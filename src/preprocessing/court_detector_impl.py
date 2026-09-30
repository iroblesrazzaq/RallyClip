import sys
import math
import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

def _load_yolo(model_path: str):
    """Load the person detector used for clean-frame extraction.

    Resolution order:
      * a pose **manifest** (``.json``, or a directory holding ``manifest.json``)
        resolves through ``load_pose_backend`` to the sha-verified ONNX model and
        runs it torch-free on the CPU EP. Clean-frame extraction only touches a
        handful of frames, so the faithful dynamic CPU model is the right call
        over the ANE static sibling (and CoreML is refused for e2e heads anyway).
      * ``.onnx`` weights use the onnxruntime runner directly.
      * anything else goes through ultralytics, which is optional at runtime.
    Returns None when no backend is available so court detection degrades to its
    single-frame fallback, as before.
    """
    p = str(model_path)
    lower = p.lower()
    if lower.endswith(".json") or Path(p).is_dir():
        from extraction.pose_backend import load_pose_backend

        model, _meta = load_pose_backend(p, provider="cpu")
        return model
    if lower.endswith(".onnx"):
        from extraction.yolo_onnx_runner import YOLO as OnnxYOLO

        return OnnxYOLO(p)
    try:
        from ultralytics import YOLO
    except ImportError:
        logging.warning("YOLO not available. Install ultralytics and ensure yolov8n.pt exists.")
        return None
    return YOLO(p)


class CourtDetector:
    """
    A class for detecting tennis court boundaries and estimating playable areas from video frames.
    """
    
    def __init__(self, yolo_model_path: str = 'models/yolov8n-pose.pt', conf: float = 0.25, device: Optional[str] = None):
        """
        Initialize the CourtDetector.

        Args:
            yolo_model_path: Path to the YOLO model file
            conf: Person-detection confidence threshold (unified with the pose pipeline conf).
        """
        self.MIN_BASELINE_LEN = 500
        # A baseline candidate must reach this fraction of the widest candidate's width to
        # count as court-spanning (filters out short floor/carpet edges). Only new tunable.
        self.BASELINE_WIDTH_RATIO = 0.6
        # --- clean-frame reconstruction (middle-anchored, outward homography) ---
        # Anchor the clean frame at the MIDDLE of the video, then grow the
        # reference search OUTWARD in CLEAN_FRAME_STEP_S hops each way, repainting
        # player-occluded pixels from homography-aligned neighbours. A side stops
        # once its homography with the base drops below CLEAN_FRAME_MIN_IOU (a
        # camera cut / large pan-zoom means the neighbour no longer describes the
        # same court) or the video runs out; the whole search stops early once no
        # occluded pixel remains (nobody left standing over the lines).
        # CLEAN_FRAME_MAX_OFFSET_S caps the reach on very long clips.
        self.CLEAN_FRAME_STEP_S = 10.0
        self.CLEAN_FRAME_MIN_IOU = 0.6
        self.CLEAN_FRAME_MAX_OFFSET_S = 180.0
        self.CLEAN_FRAME_MIN_INLIERS = 10
        self.yolo_model_path = yolo_model_path
        self.conf = float(conf)
        self.device = device
        self.yolo_model = None
        
        try:
            self.yolo_model = _load_yolo(yolo_model_path)
            if self.yolo_model is not None:
                if self.device:
                    self.yolo_model.to(self.device)
                logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.warning("YOLO model failed to load: %s", e)
            self.yolo_model = None
    
    def _detect_person_boxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Person boxes (x, y, w, h) above ``self.conf`` in one frame.

        The pose backend is single-class (person), so every detection is a person;
        the ``cls == 0`` guard keeps parity if a multi-class detector is ever wired
        in. Empty list when no model is loaded."""
        if self.yolo_model is None:
            return []
        results = self.yolo_model.predict(source=frame, verbose=False)[0]
        boxes: List[Tuple[int, int, int, int]] = []
        for box in getattr(results, "boxes", []):
            try:
                if int(box.cls.item()) != 0:
                    continue
                if float(box.conf.item()) <= self.conf:
                    continue
                x0, y0, x1, y1 = [int(v) for v in box.xyxy.cpu().numpy().reshape(-1)]
                boxes.append((x0, y0, x1 - x0, y1 - y0))
            except Exception:
                continue
        return boxes

    @staticmethod
    def _boxes_to_mask(boxes: List[Tuple[int, int, int, int]], shape: Tuple[int, int], dilate_px: int = 5) -> np.ndarray:
        """Filled 255-on-0 occlusion mask for a list of (x, y, w, h) boxes."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for x, y, w, h in boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        if dilate_px:
            mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8), iterations=1)
        return mask

    @staticmethod
    def _quad_iou(M: np.ndarray, ref_shape: Tuple[int, int], base_shape: Tuple[int, int]) -> float:
        """IoU of the reference image quad (warped into base coords by M) with the
        base rectangle. ~1.0 for a static camera, falling as it pans/zooms — the
        signal used to decide a neighbour is too far to trust as a reference."""
        try:
            rh, rw = ref_shape[:2]
            bh, bw = base_shape[:2]
            ref_quad = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(ref_quad, M).reshape(-1, 2).astype(np.float32)
            if not np.all(np.isfinite(warped)):
                return 0.0
            base_quad = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
            inter, _ = cv2.intersectConvexConvex(warped, base_quad)
            area_ref = abs(cv2.contourArea(warped))
            union = area_ref + float(bw * bh) - inter
            return float(inter / union) if union > 0 else 0.0
        except cv2.error:
            return 0.0

    def _homography_to_base(
        self,
        orb,
        kp_base,
        des_base,
        reference_frame: np.ndarray,
        base_shape: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], float, int]:
        """RANSAC homography mapping ``reference_frame`` -> base, with its coverage
        IoU and inlier count. M is None when it can't be estimated reliably."""
        kp_ref, des_ref = orb.detectAndCompute(reference_frame, None)
        if des_ref is None or des_base is None:
            return None, 0.0, 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des_ref, des_base), key=lambda m: m.distance)
        good = matches[: min(100, len(matches))]
        if len(good) < self.CLEAN_FRAME_MIN_INLIERS:
            return None, 0.0, 0
        ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        base_pts = np.float32([kp_base[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inliers = cv2.findHomography(ref_pts, base_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None, 0.0, 0
        n_inliers = int(inliers.sum()) if inliers is not None else 0
        if n_inliers < self.CLEAN_FRAME_MIN_INLIERS:
            return None, 0.0, n_inliers
        return M, self._quad_iou(M, reference_frame.shape, base_shape), n_inliers

    def extract_clean_frame(self, video_path: str, target_time: Optional[float] = None) -> np.ndarray:
        """Reconstruct a player-free frame anchored at the MIDDLE of the video.

        Takes the base frame at ``target_time`` (the video midpoint when None),
        masks the players occluding it, then walks OUTWARD in
        ``CLEAN_FRAME_STEP_S`` hops each direction, homography-aligning each
        neighbour to the base and repainting still-occluded pixels from it —
        provided that neighbour has nobody standing over the same spot. A
        direction is abandoned once its homography IoU with the base drops below
        ``CLEAN_FRAME_MIN_IOU`` (camera cut / big pan-zoom) or the video ends; the
        walk stops early the moment every occluded pixel has been repaired.

        Args:
            video_path: Path to the video file.
            target_time: Anchor time in seconds; None => middle of the video.

        Returns:
            np.ndarray: Clean frame with player occlusions repainted where possible.
        """
        from runtime.video_frames import VideoFrameReader

        reader = VideoFrameReader(video_path)
        fps = reader.fps
        total_frames = reader.total_frames
        duration = (total_frames / fps) if fps else 0.0

        try:
            # Anchor at the middle unless a caller pins a time (the court
            # regression goldens do). Mid-match is the most reliable place to find
            # in-play footage with a settled camera.
            if target_time is None:
                target_time = duration / 2.0 if duration > 0 else 60.0

            if self.yolo_model is None:
                logging.info("YOLO not available, using single frame at target time")
                frame = reader.read_frame_at_index(int(fps * target_time))
                if frame is None:
                    raise RuntimeError("Could not read frame at target time.")
                return frame

            base_frame = reader.read_frame_at_index(int(target_time * fps))
            if base_frame is None:
                raise RuntimeError("Could not read base frame at target time.")
            h, w = base_frame.shape[:2]

            occlusion = self._boxes_to_mask(self._detect_person_boxes(base_frame), base_frame.shape)
            total_occ = int(np.count_nonzero(occlusion))
            if total_occ == 0:
                logging.info("No players occlude the base frame at %.1fs; using it as-is", target_time)
                return base_frame

            orb = cv2.ORB_create(nfeatures=1000)
            kp_base, des_base = orb.detectAndCompute(base_frame, None)
            if des_base is None:
                logging.info("No features in base frame; cannot repair occlusions")
                return base_frame

            clean = base_frame.copy()
            remaining = occlusion.copy()
            step = max(1.0, self.CLEAN_FRAME_STEP_S)
            max_off = min(self.CLEAN_FRAME_MAX_OFFSET_S, max(duration - target_time, target_time))
            stopped = {1: False, -1: False}
            n_refs = 0
            offset = step
            while offset <= max_off and not (stopped[1] and stopped[-1]):
                for sign in (1, -1):
                    if stopped[sign]:
                        continue
                    t = target_time + sign * offset
                    if t < 0 or t * fps >= total_frames:
                        stopped[sign] = True
                        continue
                    ref = reader.read_frame_at_index(int(t * fps))
                    if ref is None:
                        stopped[sign] = True
                        continue
                    M, iou, n_in = self._homography_to_base(orb, kp_base, des_base, ref, base_frame.shape)
                    if M is None or iou < self.CLEAN_FRAME_MIN_IOU:
                        logging.debug(
                            "Clean-frame ref t=%.1fs rejected (iou=%.2f inliers=%s)", t, iou, n_in
                        )
                        stopped[sign] = True
                        continue
                    warped = cv2.warpPerspective(ref, M, (w, h))
                    coverage = cv2.warpPerspective(np.full((ref.shape[0], ref.shape[1]), 255, np.uint8), M, (w, h))
                    ref_occ_warped = cv2.warpPerspective(
                        self._boxes_to_mask(self._detect_person_boxes(ref), ref.shape), M, (w, h)
                    )
                    # Repaint pixels that are still occluded in the base, are
                    # covered by this warped neighbour, and are NOT under a player
                    # in that neighbour.
                    fillable = (remaining > 0) & (coverage > 0) & (ref_occ_warped == 0)
                    if np.any(fillable):
                        clean[fillable] = warped[fillable]
                        remaining[fillable] = 0
                        n_refs += 1
                    if not np.any(remaining):
                        break
                if not np.any(remaining):
                    break
                offset += step

            repaired = total_occ - int(np.count_nonzero(remaining))
            logging.info(
                "Clean frame @%.1fs: repaired %d/%d occluded px from %d reference frame(s)%s",
                target_time, repaired, total_occ, n_refs,
                "" if not np.any(remaining) else " (residual occlusion remains)",
            )
            return clean.astype(np.uint8)
        finally:
            reader.close()
    
    def detect_court_lines(self, frame: np.ndarray) -> Tuple[List, List, List, List]:
        """
        Detect court lines from a frame using edge detection and line detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines, right_diagonals, left_diagonals)
        """
        # Pre-processing for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Generate the two initial masks
        # Find all high-contrast edges with Canny
        canny_edges = cv2.Canny(blurred, 50, 150)
        
        # Find all white pixels using LAB color space
        lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lower_white = np.array([145, 105, 105])
        upper_white = np.array([255, 150, 150])
        white_mask = cv2.inRange(lab_image, lower_white, upper_white)
        
        # Create the "proximity mask" from Canny edges
        kernel = np.ones((4,4), np.uint8)
        dilated_edges_mask = cv2.dilate(canny_edges, kernel, iterations=1)
        
        # Find the intersection to get the refined mask
        refined_lines_mask = cv2.bitwise_and(white_mask, dilated_edges_mask)
        refined_lines_mask = cv2.morphologyEx(refined_lines_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply ROI masking
        height, width = frame.shape[:2]
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define the cutoff points for the top
        top_cutoff = int(height * 0.35)
        
        # Define the vertices of the polygon to keep
        roi_vertices = np.array([
            (0, height),                                  # Bottom-left
            (0, top_cutoff),                             # Left edge
            (width, top_cutoff),                         # Right edge
            (width, height)                              # Bottom-right
        ], dtype=np.int32)
        
        # Fill the polygon area
        cv2.fillPoly(roi_mask, [roi_vertices], 255)
        
        # Apply ROI mask
        masked_result = cv2.bitwise_and(refined_lines_mask, refined_lines_mask, mask=roi_mask)
        
        # Detect lines using Hough Transform
        linesP = cv2.HoughLinesP(masked_result, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=40)

        if linesP is None:
            return [], [], [], []
        # OpenCV 4 returns shape (N, 1, 4); OpenCV 5 dropped the middle axis.
        # Normalize so every downstream `line[0]` unpack works on both.
        linesP = np.asarray(linesP).reshape(-1, 1, 4)
        
        # Classify lines
        screen_center_x = frame.shape[1] / 2
        horizontal_lines, vertical_lines, right_diagonals, left_diagonals = [], [], [], []
        
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            _, angle_deg = self._get_polar_angle(line[0])
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Normalize the angle to a [0, 180) degree range
            normalized_angle = int(angle_deg % 180)
            
            # Determine if the line is on the Left or Right side
            side_label = "L" if mid_x < screen_center_x else "R"
            
            # Classify lines
            if normalized_angle < 15 or normalized_angle > 165:  # Horizontal
                horizontal_lines.append(line)
            elif 75 < normalized_angle < 105:  # Vertical
                vertical_lines.append(line)
            elif 15 <= normalized_angle <= 75:  # Positive Slope
                if side_label == "R":
                    right_diagonals.append(line)
            elif 105 <= normalized_angle <= 165:  # Negative Slope
                if side_label == "L":
                    left_diagonals.append(line)
        
        return horizontal_lines, vertical_lines, right_diagonals, left_diagonals
    
    def merge_lines(self, lines: List, image_shape: Tuple[int, int], 
                   kernel_size: Tuple[int, int] = (5, 25), 
                   iterations: int = 2, 
                   min_contour_area: int = 50) -> List:
        """
        Merge line segments by drawing them on a mask and using morphology.
        
        Args:
            lines: List of lines to merge
            image_shape: Shape of the image (height, width)
            kernel_size: Kernel size for morphological operations
            iterations: Number of iterations for morphological operations
            min_contour_area: Minimum contour area to keep
            
        Returns:
            List of merged lines
        """
        if not lines:
            return []
        
        # Create a blank mask for the initial drawing
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
        
        # Use morphological CLOSE operation to connect nearby segments
        kernel = np.ones(kernel_size, np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # Find the contours of the connected blobs
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_merged_lines = []
        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            
            max_dist = 0
            p1_final, p2_final = None, None
            # The farthest pair always lies on the convex hull, so scan the
            # hull instead of every contour point.
            points = cv2.convexHull(contour).reshape(-1, 2)

            for p1 in points:
                for p2 in points:
                    dist = np.linalg.norm(p1 - p2)
                    if dist > max_dist:
                        max_dist = dist
                        p1_final, p2_final = p1, p2
            
            if p1_final is not None:
                final_line = [[int(p1_final[0]), int(p1_final[1]), int(p2_final[0]), int(p2_final[1])]]
                final_merged_lines.append(final_line)
        
        return final_merged_lines
    
    def find_baseline(self, horizontal_lines: List) -> Optional[List]:
        """
        Find the baseline from horizontal lines.

        Among the lines long enough to be a baseline, restrict to the genuinely
        court-spanning ones (>= BASELINE_WIDTH_RATIO of the widest candidate) before
        taking the lowest. This stops a short floor/carpet edge near the bottom of the
        frame from beating the real, wider baseline that sits slightly higher.

        Args:
            horizontal_lines: List of horizontal lines

        Returns:
            The baseline line or None if not found
        """
        candidates = [
            line for line in horizontal_lines
            if abs(line[0][2] - line[0][0]) >= self.MIN_BASELINE_LEN  # long enough for baseline
        ]
        if not candidates:
            return None
        max_width = max(abs(line[0][2] - line[0][0]) for line in candidates)
        court_spanning = [
            line for line in candidates
            if abs(line[0][2] - line[0][0]) >= self.BASELINE_WIDTH_RATIO * max_width
        ]
        # Lowest (greatest mean-y) of the court-spanning lines: prefers the front baseline.
        return max(court_spanning, key=lambda line: (line[0][1] + line[0][3]) / 2)
    
    def process_side_decision_tree(self, diagonal_lines: List, baseline: List, 
                                 image_width: int, side: str) -> Optional[List]:
        """
        Process one side of the court using decision tree logic.
        
        Args:
            diagonal_lines: List of diagonal lines for one side
            baseline: The baseline line
            image_width: Width of the image
            side: "left" or "right" to indicate which side
            
        Returns:
            The doubles sideline or None if not found
        """
        line_count = len(diagonal_lines)
        
        if line_count == 0:
            logging.debug("%s side: No diagonal lines found", side.capitalize())
            return None
        elif line_count == 1:
            logging.debug("%s side: Only one diagonal line found", side.capitalize())
            return None
        elif line_count == 2:
            logging.debug("%s side: Ideal case (2 lines)", side.capitalize())
            doubles_sideline = self._find_outer_line(diagonal_lines)
            if self._validate_sideline_candidate(doubles_sideline, baseline, image_width):
                return doubles_sideline
            else:
                logging.debug("%s side: Validation failed for ideal case", side.capitalize())
                return None
        else:
            logging.debug("%s side: Red herring case (%s lines)", side.capitalize(), line_count)
            
            # Check baseline width to determine strategy
            bx1, by1, bx2, by2 = baseline[0]
            baseline_width = abs(bx2 - bx1)
            baseline_width_percentage = (baseline_width / image_width) * 100
            
            if baseline_width_percentage > 98.5:
                # Full-width baseline case
                doubles_sideline, is_valid = self._process_full_width_baseline_case(
                    diagonal_lines, baseline, image_width, side)
                if is_valid:
                    return doubles_sideline
                else:
                    logging.debug("%s side: Full-width baseline case failed", side.capitalize())
                    return None
            else:
                # Partially visible baseline
                doubles_sideline = self._process_partial_baseline_case(
                    diagonal_lines, baseline, image_width, side)
                if self._validate_sideline_candidate(doubles_sideline, baseline, image_width):
                    return doubles_sideline
                else:
                    logging.debug("%s side: Partial baseline case failed", side.capitalize())
                    return None
    
    def estimate_playable_court_area(self, left_doubles_sideline: Optional[List], 
                                   right_doubles_sideline: Optional[List], 
                                   baseline: Optional[List], 
                                   image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create an "out" mask, where white pixels represent areas outside the playable court.
        
        Args:
            left_doubles_sideline: The coordinates of the detected left sideline [(x1, y1, x2, y2)]
            right_doubles_sideline: The coordinates of the detected right sideline [(x1, y1, x2, y2)]
            baseline: The coordinates of the detected baseline [(x1, y1, x2, y2)]
            image_shape: Tuple of (height, width) of the video frame
        
        Returns:
            numpy.ndarray: Binary mask where white (255) represents the "out" area.
        """
        if left_doubles_sideline is None or right_doubles_sideline is None or baseline is None:
            logging.warning("Missing court lines, cannot estimate 'out' area mask")
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        # --- Calculate the EXTENDED sidelines (same logic as in draw_court_lines) ---
        BASE_HORIZONTAL_SHIFT = 100
        screen_width = image_shape[1]
        bx1, by1, bx2, by2 = baseline[0]
        baseline_width = abs(bx2 - bx1)
        scale_factor = (baseline_width / screen_width)
        dynamic_shift = BASE_HORIZONTAL_SHIFT * scale_factor
        
        # Get line equations for original sidelines
        lx1, ly1, lx2, ly2 = left_doubles_sideline[0]
        left_slope = (ly2 - ly1) / (lx2 - lx1) if (lx2 - lx1) != 0 else float('inf')
        left_intercept = ly1 - left_slope * lx1
        
        rx1, ry1, rx2, ry2 = right_doubles_sideline[0]
        right_slope = (ry2 - ry1) / (rx2 - rx1) if (rx2 - rx1) != 0 else float('inf')
        right_intercept = ry1 - right_slope * rx1
        
        # Calculate shifted intercepts for the extended sidelines (pink and yellow lines)
        if left_slope != float('inf'):
            left_shifted_intercept = left_intercept - dynamic_shift / np.sqrt(1 + left_slope**2)
        else:
            left_shifted_intercept = left_intercept  # Not used for vertical line, but for completeness
        
        if right_slope != float('inf'):
            right_shifted_intercept = right_intercept - dynamic_shift / np.sqrt(1 + right_slope**2)
        else:
            right_shifted_intercept = right_intercept # Not used for vertical line
        
        # --- Create the 'Out' Mask ---
        height, width = image_shape[:2]
        # Initialize a black mask. We will add the white 'out' areas to it.
        out_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create a coordinate grid for the entire frame
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # 1. Process EXTENDED left sideline (pink line)
        if left_slope != float('inf'):
            # Calculate the y-value of the line for every x-coordinate in the frame
            extended_left_y_values = left_slope * x_coords + left_shifted_intercept
            # The "out" area is to the left, which is ABOVE the line in the image (smaller y-values)
            left_out_area = y_coords < extended_left_y_values
            out_mask[left_out_area] = 255
        else:
            # Vertical line case
            left_shifted_x = lx1 - dynamic_shift
            out_mask[x_coords < left_shifted_x] = 255
        
        # 2. Process EXTENDED right sideline (yellow line)
        if right_slope != float('inf'):
            # Calculate the y-value of the line for every x-coordinate in the frame
            extended_right_y_values = right_slope * x_coords + right_shifted_intercept
            # The "out" area is to the right, which is ALSO ABOVE the line in the image (smaller y-values)
            right_out_area = y_coords < extended_right_y_values
            out_mask[right_out_area] = 255
        else:
            # Vertical line case
            right_shifted_x = rx1 + dynamic_shift
            out_mask[x_coords > right_shifted_x] = 255
        
        return out_mask
    
    def process_video(self, video_path: str, target_time: Optional[float] = None) -> Tuple[Optional[np.ndarray], np.ndarray, dict]:
        """
        Main wrapper method to process a video and return the "out" mask.

        Args:
            video_path: Path to the video file
            target_time: Anchor time in seconds for the clean frame; None => the
                middle of the video (the new default).

        Returns:
            Tuple of (out_mask, clean_frame, metadata) where:
            - out_mask is a binary mask where white (255) represents areas outside the playable court
            - out_mask is None if court detection failed
            - clean_frame is the extracted frame
            - metadata contains detection status and error information
        """
        logging.info("Processing video: %s", video_path)
        
        # Initialize metadata
        metadata = {
            "court_detection_success": False,
            "error": None,
            "baseline_found": False,
            "left_sideline_found": False,
            "right_sideline_found": False,
            "baseline_width": 0,
            "image_width": 0
        }
        
        try:
            # Step 1: Extract clean frame
            clean_frame = self.extract_clean_frame(video_path, target_time)
            metadata["image_width"] = clean_frame.shape[1]
            
            # Step 2: Detect court lines
            horizontal_lines, vertical_lines, right_diagonals, left_diagonals = self.detect_court_lines(clean_frame)
            
            # Step 3: Merge lines
            merged_horizontal = self.merge_lines(horizontal_lines, clean_frame.shape, kernel_size=(5, 30))
            merged_right_diagonals = self.merge_lines(right_diagonals, clean_frame.shape, kernel_size=(2, 2))
            merged_left_diagonals = self.merge_lines(left_diagonals, clean_frame.shape, kernel_size=(2, 2))
            
            # Step 4: Find baseline
            baseline = self.find_baseline(merged_horizontal)
            if baseline is None:
                logging.warning("No valid baseline found - court detection failed")
                metadata["error"] = "No baseline found"
                return None, clean_frame, metadata
            
            metadata["baseline_found"] = True
            metadata["baseline_width"] = abs(baseline[0][2] - baseline[0][0])
            
            # Step 5: Process sides
            right_doubles_sideline = self.process_side_decision_tree(
                merged_right_diagonals, baseline, clean_frame.shape[1], "right")
            left_doubles_sideline = self.process_side_decision_tree(
                merged_left_diagonals, baseline, clean_frame.shape[1], "left")
            
            metadata["left_sideline_found"] = left_doubles_sideline is not None
            metadata["right_sideline_found"] = right_doubles_sideline is not None
            
            # Check if we have enough court lines for a valid mask
            if not left_doubles_sideline or not right_doubles_sideline:
                logging.warning("Missing sidelines - court detection failed")
                if not left_doubles_sideline and not right_doubles_sideline:
                    metadata["error"] = "Both sidelines missing"
                elif not left_doubles_sideline:
                    metadata["error"] = "Left sideline missing"
                else:
                    metadata["error"] = "Right sideline missing"
                return None, clean_frame, metadata
            
            # Step 6: Generate the "out" mask
            out_mask = self.estimate_playable_court_area(left_doubles_sideline, right_doubles_sideline, baseline, clean_frame.shape)
            
            if out_mask is None or not np.any(out_mask):
                logging.warning("Failed to generate valid court mask")
                metadata["error"] = "Failed to generate court mask"
                return None, clean_frame, metadata
            
            # Success!
            metadata["court_detection_success"] = True
            metadata["error"] = None
            logging.info("Court detection successful")
            
            return out_mask, clean_frame, metadata
            
        except Exception as e:
            logging.error("Court detection failed with exception: %s", e, exc_info=True)
            metadata["error"] = f"Exception during court detection: {str(e)}"
            # If clean_frame wasn't created yet, create a dummy one
            if 'clean_frame' not in locals():
                clean_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            return None, clean_frame, metadata
    
    def draw_court_lines(self, frame: np.ndarray, baseline: Optional[List], 
                        left_doubles_sideline: Optional[List], 
                        right_doubles_sideline: Optional[List]) -> np.ndarray:
        """
        Draw the detected court lines on the frame, similar to manual_court2.py.
        
        Args:
            frame: Input frame
            baseline: The baseline line
            left_doubles_sideline: The left doubles sideline
            right_doubles_sideline: The right doubles sideline
            
        Returns:
            Frame with court lines drawn
        """
        final_result_image = frame.copy()
        failures = []
        
        # Draw baseline (green)
        if baseline is not None:
            cv2.line(final_result_image, (baseline[0][0], baseline[0][1]), 
                     (baseline[0][2], baseline[0][3]), (0, 255, 0), 3)
        
        # Draw doubles sidelines (if found) - extended through the whole image
        if right_doubles_sideline is not None:
            # Get line equation for right sideline
            rx1, ry1, rx2, ry2 = right_doubles_sideline[0]
            right_slope = (ry2 - ry1) / (rx2 - rx1) if (rx2 - rx1) != 0 else float('inf')
            right_intercept = ry1 - right_slope * rx1
            
            # Calculate endpoints at image boundaries
            if right_slope != float('inf'):
                # Calculate y at x=0 and x=image_width
                right_y_at_x0 = int(right_slope * 0 + right_intercept)
                right_y_at_xmax = int(right_slope * frame.shape[1] + right_intercept)
                cv2.line(final_result_image, (0, right_y_at_x0), (frame.shape[1], right_y_at_xmax), (255, 0, 0), 5)
            else:
                # Vertical line
                cv2.line(final_result_image, (rx1, 0), (rx1, frame.shape[0]), (255, 0, 0), 5)
            logging.info("Right doubles sideline: FOUND")
        else:
            logging.info("Right doubles sideline: NOT FOUND")
            failures.append("RIGHT SIDELINE")
            
        if left_doubles_sideline is not None:
            # Get line equation for left sideline
            lx1, ly1, lx2, ly2 = left_doubles_sideline[0]
            left_slope = (ly2 - ly1) / (lx2 - lx1) if (lx2 - lx1) != 0 else float('inf')
            left_intercept = ly1 - left_slope * lx1
            
            # Calculate endpoints at image boundaries
            if left_slope != float('inf'):
                # Calculate y at x=0 and x=image_width
                left_y_at_x0 = int(left_slope * 0 + left_intercept)
                left_y_at_xmax = int(left_slope * frame.shape[1] + left_intercept)
                cv2.line(final_result_image, (0, left_y_at_x0), (frame.shape[1], left_y_at_xmax), (0, 0, 255), 5)
            else:
                # Vertical line
                cv2.line(final_result_image, (lx1, 0), (lx1, frame.shape[0]), (0, 0, 255), 5)
            logging.info("Left doubles sideline: FOUND")
        else:
            logging.info("Left doubles sideline: NOT FOUND")
            failures.append("LEFT SIDELINE")
        
        # Draw extended doubles sidelines in pink and yellow (if both sidelines are found)
        if left_doubles_sideline is not None and right_doubles_sideline is not None and baseline is not None:
            # Calculate shifted sidelines for visualization
            BASE_HORIZONTAL_SHIFT = 100
            screen_width = frame.shape[1]
            bx1, by1, bx2, by2 = baseline[0]
            baseline_width = abs(bx2 - bx1)
            scale_factor = baseline_width / screen_width
            dynamic_shift = BASE_HORIZONTAL_SHIFT * scale_factor
            
            # Draw extended sidelines in pink and yellow - extended through the whole image
            # Get line equations for extended sidelines
            lx1, ly1, lx2, ly2 = left_doubles_sideline[0]
            left_slope = (ly2 - ly1) / (lx2 - lx1) if (lx2 - lx1) != 0 else float('inf')
            left_intercept = ly1 - left_slope * lx1
            
            rx1, ry1, rx2, ry2 = right_doubles_sideline[0]
            right_slope = (ry2 - ry1) / (rx2 - rx1) if (rx2 - rx1) != 0 else float('inf')
            right_intercept = ry1 - right_slope * rx1
            
            # Calculate shifted intercepts with proper outward direction
            if left_slope != float('inf'):
                # For left sideline, shift outward (away from center of image)
                # Left sideline should be shifted to the left (negative x direction)
                left_shifted_intercept = left_intercept - dynamic_shift / np.sqrt(1 + left_slope**2)
                left_y_at_x0 = int(left_slope * 0 + left_shifted_intercept)
                left_y_at_xmax = int(left_slope * frame.shape[1] + left_shifted_intercept)
                cv2.line(final_result_image, (0, left_y_at_x0), (frame.shape[1], left_y_at_xmax), (147, 20, 255), 3)  # Pink
            else:
                # Vertical line shifted horizontally (leftward for left sideline)
                left_shifted_x = lx1 - dynamic_shift
                cv2.line(final_result_image, (left_shifted_x, 0), (left_shifted_x, frame.shape[0]), (147, 20, 255), 3)  # Pink
            
            if right_slope != float('inf'):
                # For right sideline, shift outward (away from center of image)
                # Right sideline should be shifted up (decrease y-intercept)
                right_shifted_intercept = right_intercept - dynamic_shift / np.sqrt(1 + right_slope**2)
                
                right_y_at_x0 = int(right_slope * 0 + right_shifted_intercept)
                right_y_at_xmax = int(right_slope * frame.shape[1] + right_shifted_intercept)
                cv2.line(final_result_image, (0, right_y_at_x0), (frame.shape[1], right_y_at_xmax), (0, 255, 255), 3)  # Yellow
            else:
                # Vertical line shifted horizontally (rightward for right sideline)
                right_shifted_x = rx1 + dynamic_shift
                cv2.line(final_result_image, (right_shifted_x, 0), (right_shifted_x, frame.shape[0]), (0, 255, 255), 3)  # Yellow
            
            logging.info(
                "Extended doubles sidelines drawn (left: pink, right: yellow, shift: %.1fpx)",
                dynamic_shift,
            )
        
        # Add failure text to image
        if failures:
            # Set up text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_color = (0, 0, 255)  # Red for failures
            font_thickness = 2
            
            # Create failure message
            failure_text = "FAILED: " + ", ".join(failures)
            
            # Get text size for positioning
            (text_width, text_height), baseline_text = cv2.getTextSize(failure_text, font, font_scale, font_thickness)
            
            # Position text in top-left corner with some padding
            text_x = 20
            text_y = 40
            
            # Add black background rectangle for better visibility
            cv2.rectangle(final_result_image, 
                         (text_x - 10, text_y - text_height - 10),
                         (text_x + text_width + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # Add the failure text
            cv2.putText(final_result_image, failure_text, (text_x, text_y), 
                       font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        return final_result_image
    
    # Helper methods
    def _get_polar_angle(self, line: List[int]) -> Tuple[float, float]:
        """Calculate the polar angle of a line given its endpoints."""
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        return angle_rad, angle_deg
    
    def _find_outer_line(self, lines: List) -> Optional[List]:
        """Find the outer line (lower y-coordinate) from a list of lines."""
        if len(lines) < 2:
            return lines[0] if lines else None
        
        line1, line2 = lines[0], lines[1]
        
        # Try midpoint of line1
        x1, y1, x2, y2 = line1[0]
        mid_x1 = (x1 + x2) / 2
        
        if self._is_x_in_line_domain(line2, mid_x1):
            y1_at_mid = self._get_y_at_x(line1, mid_x1)
            y2_at_mid = self._get_y_at_x(line2, mid_x1)
            if y1_at_mid is not None and y2_at_mid is not None:
                return line1 if y1_at_mid < y2_at_mid else line2
        
        # Try midpoint of line2
        x1, y1, x2, y2 = line2[0]
        mid_x2 = (x1 + x2) / 2
        
        if self._is_x_in_line_domain(line1, mid_x2):
            y1_at_mid = self._get_y_at_x(line1, mid_x2)
            y2_at_mid = self._get_y_at_x(line2, mid_x2)
            if y1_at_mid is not None and y2_at_mid is not None:
                return line1 if y1_at_mid < y2_at_mid else line2
        
        # If no shared x-point found, use the line with lower average y-coordinate
        avg_y1 = (line1[0][1] + line1[0][3]) / 2
        avg_y2 = (line2[0][1] + line2[0][3]) / 2
        return line1 if avg_y1 < avg_y2 else line2
    
    def _validate_sideline_candidate(self, candidate: Optional[List], baseline: Optional[List], 
                                   image_width: int) -> bool:
        """Check if candidate line is close enough to baseline."""
        if candidate is None or baseline is None:
            return False
        
        # Calculate baseline width as percentage of screen width
        bx1, by1, bx2, by2 = baseline[0]
        baseline_width = abs(bx2 - bx1)
        baseline_width_percentage = (baseline_width / image_width) * 100
        
        # Adjust tolerance based on baseline width
        if baseline_width_percentage <= 98.5:
            tolerance = 100  # Baseline is mostly visible
        else:
            tolerance = 150  # Baseline is cut off, doubles sidelines might not reach it
        
        # Get the bottom endpoint of the candidate (higher y-coordinate)
        x1, y1, x2, y2 = candidate[0]
        candidate_bottom_y = max(y1, y2)
        
        # Get the y-coordinate of the baseline
        baseline_y = (by1 + by2) / 2
        
        # Check if the candidate's bottom is close to the baseline
        return abs(candidate_bottom_y - baseline_y) <= tolerance
    
    def _process_full_width_baseline_case(self, diagonal_lines: List, baseline: List, 
                                        image_width: int, side: str) -> Tuple[Optional[List], bool]:
        """Process the case where baseline is full-width (>98.5%) and there are more than 2 diagonal lines."""
        bx1, by1, bx2, by2 = baseline[0]
        baseline_y = (by1 + by2) / 2
        tolerance = 100  # pixels
        
        # Find lines that are vertically close to the baseline
        close_lines = []
        for line in diagonal_lines:
            x1, y1, x2, y2 = line[0]
            lowest_y = max(y1, y2)
            if abs(lowest_y - baseline_y) < tolerance:
                close_lines.append(line)
        
        logging.info(
            "%s side: Found %s lines close to baseline out of %s total",
            side.capitalize(),
            len(close_lines),
            len(diagonal_lines),
        )
        
        # Decision tree based on number of close lines
        if len(close_lines) == 1:
            # Branch 1: Exactly ONE line is close to baseline
            logging.info("%s side: Branch 1 - One close line (assumed singles sideline)", side.capitalize())
            
            singles_line = close_lines[0]
            far_lines = [line for line in diagonal_lines if line not in close_lines]
            
            # Get midpoint of the assumed singles line
            sx1, sy1, sx2, sy2 = singles_line[0]
            mid_x = (sx1 + sx2) / 2
            mid_y = (sy1 + sy2) / 2
            
            # Calculate reference x
            slope_singles, y_intercept_singles = self._get_line_equation(singles_line)
            if slope_singles is None:  # vertical line
                x_reference = sx1
            else:
                x_reference = (mid_y - y_intercept_singles) / slope_singles
            
            # Find the best candidate from far lines
            best_candidate = None
            min_distance = float('inf')
            
            for line in far_lines:
                x1, y1, x2, y2 = line[0]
                slope_candidate, y_intercept_candidate = self._get_line_equation(line)
                
                if slope_candidate is None:  # vertical line
                    x_candidate = x1
                else:
                    x_candidate = (mid_y - y_intercept_candidate) / slope_candidate
                
                # Check for "outwardness"
                if side == "right" and x_candidate > x_reference:
                    distance = abs(x_candidate - x_reference)
                    if distance < min_distance:
                        min_distance = distance
                        best_candidate = line
                elif side == "left" and x_candidate < x_reference:
                    distance = abs(x_candidate - x_reference)
                    if distance < min_distance:
                        min_distance = distance
                        best_candidate = line
            
            if best_candidate is not None:
                return best_candidate, True
            else:
                logging.info("%s side: No valid outward candidate found", side.capitalize())
                return None, False
                
        elif len(close_lines) == 2:
            # Branch 2: Exactly TWO lines are close to baseline
            logging.info("%s side: Branch 2 - Two close lines (singles and doubles sidelines)", side.capitalize())
            
            line1, line2 = close_lines[0], close_lines[1]
            
            # Get midpoint of first line
            x1, y1, x2, y2 = line1[0]
            mid_x = (x1 + x2) / 2
            
            # Check if mid_x is within domain of second line
            x3, y3, x4, y4 = line2[0]
            if not (min(x3, x4) <= mid_x <= max(x3, x4)):
                mid_x = (x3 + x4) / 2
            
            # Calculate y-coordinates for both lines at shared mid_x
            slope1, y_intercept1 = self._get_line_equation(line1)
            slope2, y_intercept2 = self._get_line_equation(line2)
            
            if slope1 is None:  # vertical line
                y1_at_mid = y1
            else:
                y1_at_mid = slope1 * mid_x + y_intercept1
                
            if slope2 is None:  # vertical line
                y2_at_mid = y3
            else:
                y2_at_mid = slope2 * mid_x + y_intercept2
            
            # Select the outer line (lower y-coordinate = higher on screen)
            if y1_at_mid < y2_at_mid:
                outer_line = line1
            else:
                outer_line = line2
            
            return outer_line, True
            
        else:
            # Branch 3: ZERO or MORE THAN TWO lines are close to baseline
            logging.info("%s side: Branch 3 - Ambiguous case (%s close lines)", side.capitalize(), len(close_lines))
            return None, False
    
    def _process_partial_baseline_case(self, diagonal_lines: List, baseline: List, 
                                     image_width: int, side: str) -> Optional[List]:
        """Process case where baseline is partially visible (≤98.5% width)."""
        bx1, by1, bx2, by2 = baseline[0]
        baseline_y = (by1 + by2) / 2
        
        if side == "right":
            baseline_end_x = max(bx1, bx2)
        else:  # left
            baseline_end_x = min(bx1, bx2)
        
        min_distance = float('inf')
        best_candidate = None
        
        for line in diagonal_lines:
            x1, y1, x2, y2 = line[0]
            # Find the end of the line closest to the baseline end
            near_end_x = x1 if y1 > y2 else x2
            near_end_y = max(y1, y2)
            distance = np.sqrt((near_end_x - baseline_end_x)**2 + (near_end_y - baseline_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_candidate = line
        
        return best_candidate
    
    def _get_line_equation(self, line: List) -> Tuple[Optional[float], float]:
        """Get slope and y-intercept for a line segment."""
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:  # vertical line
            return None, x1  # slope=None, x_intercept
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1
        return slope, y_intercept
    
    def _get_y_at_x(self, line: List, x: float) -> Optional[float]:
        """Get y-coordinate of line at given x-coordinate."""
        slope, y_intercept = self._get_line_equation(line)
        if slope is None:  # vertical line
            return None
        return slope * x + y_intercept
    
    def _is_x_in_line_domain(self, line: List, x: float) -> bool:
        """Check if x-coordinate is within the domain of the line segment."""
        x1, y1, x2, y2 = line[0]
        return min(x1, x2) <= x <= max(x1, x2)
    
    def _shift_line_perpendicular(self, line: List, shift_amount: float, outward_direction: int) -> Tuple[int, int, int, int]:
        """Shift a line perpendicular to its direction by the specified amount."""
        x1, y1, x2, y2 = line[0]
        
        # Calculate the line vector
        vx = x2 - x1
        vy = y2 - y1
        
        # Calculate perpendicular normal vector (outward from court)
        perp_x = -vy
        perp_y = vx
        
        # Normalize to get unit vector
        length = np.sqrt(perp_x**2 + perp_y**2)
        if length == 0:
            return (x1, y1, x2, y2)  # Return original if line has no length
        
        unit_perp_x = perp_x / length
        unit_perp_y = perp_y / length
        
        # Apply shift in outward direction
        shift_x = unit_perp_x * shift_amount * outward_direction
        shift_y = unit_perp_y * shift_amount * outward_direction
        
        # Calculate new endpoints
        new_x1 = int(x1 + shift_x)
        new_y1 = int(y1 + shift_y)
        new_x2 = int(x2 + shift_x)
        new_y2 = int(y2 + shift_y)
        
        return (new_x1, new_y1, new_x2, new_y2)


def filter_players_by_playable_area(player_bboxes: List[Tuple[int, int, int, int]], 
                                  playable_area_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Filter player bounding boxes to only include those within the playable area.
    
    Args:
        player_bboxes: List of player bounding boxes [(x, y, w, h), ...]
        playable_area_mask: Binary mask where white area is playable
    
    Returns:
        List: Filtered list of player bounding boxes within playable area
    """
    filtered_bboxes = []
    
    for bbox in player_bboxes:
        x, y, w, h = bbox
        
        # Calculate the center point of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Check if the center point is within the playable area
        if (0 <= center_x < playable_area_mask.shape[1] and 
            0 <= center_y < playable_area_mask.shape[0] and
            playable_area_mask[center_y, center_x] == 255):
            filtered_bboxes.append(bbox)
    
    logging.debug("Filtered %s players to %s in playable area", len(player_bboxes), len(filtered_bboxes))
    return filtered_bboxes


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manual_court.py <path_to_video>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1)

    # Initialize the court detector
    detector = CourtDetector()
    
    # Process the video and get the mask (anchored at the video midpoint)
    out_mask, clean_frame, metadata = detector.process_video(video_path, target_time=None)
    
    if out_mask is not None and np.any(out_mask):
        # Create court_masks directory if it doesn't exist
        os.makedirs("court_masks", exist_ok=True)
        
        # Save the mask with descriptive filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        mask_path = f"court_masks/{base_name}_mask.png"
        cv2.imwrite(mask_path, out_mask)
        
        print(f"\n✅ Successfully generated and saved mask for {os.path.basename(video_path)}")
        print(f"   - Mask Path: {mask_path}")
        print(f"   - Metadata: {metadata}")
        
        # Save a visualization frame for checking
        masked_frame = cv2.bitwise_and(clean_frame, clean_frame, mask=~out_mask)  # invert mask for viewing
        cv2.imwrite("court_detection_visualization.png", masked_frame)
        print("   - Visualization saved to 'court_detection_visualization.png'")
    else:
        print(f"\n❌ Failed to generate mask for {os.path.basename(video_path)}")
        print(f"   - Reason: {metadata.get('error', 'Unknown error')}")
