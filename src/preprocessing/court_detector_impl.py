import sys
import math
import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Union

# Import YOLO directly
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available. Install ultralytics and ensure yolov8n.pt exists.")


class CourtDetector:
    """
    A class for detecting tennis court boundaries and estimating playable areas from video frames.
    """
    
    def __init__(self, yolo_model_path: str = 'yolov8s.pt'):
        """
        Initialize the CourtDetector.
        
        Args:
            yolo_model_path: Path to the YOLO model file
        """
        self.MIN_BASELINE_LEN = 500
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        
        # Initialize YOLO model if available
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                logging.info("YOLO model loaded successfully")
            except Exception as e:
                logging.warning("YOLO model failed to load: %s", e)
                self.yolo_model = None
    
    def extract_clean_frame(self, video_path: str, target_time: int = 60) -> np.ndarray:
        """
        Extract a clean frame from the video at the specified time, using YOLO to remove player occlusions.
        
        Args:
            video_path: Path to the video file
            target_time: Time in seconds to extract frame from (default: 60)
            
        Returns:
            np.ndarray: Clean frame with player occlusions removed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        try:
            if self.yolo_model is None:
                # Fallback to single frame extraction
                logging.info("YOLO not available, using single frame at target time")
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * target_time))
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError("Could not read frame at target time.")
                return frame
            
            # Use YOLO + Homography for robust background reconstruction
            logging.info("Using YOLO + Homography for robust background reconstruction")
            
            # Step 1: Select base frame and find occlusions
            base_frame_num = int(target_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame_num)
            ret, base_frame = cap.read()
            if not ret or base_frame is None:
                raise RuntimeError("Could not read base frame at target time.")
            
            # Run YOLO on base frame to detect players
            results = self.yolo_model.predict(source=base_frame, verbose=False)[0]
            player_bboxes = []
            for box in getattr(results, "boxes", []):
                try:
                    cls_id = int(box.cls.item())
                    if cls_id == 0:  # person class
                        conf = float(box.conf.item())
                        if conf > 0.5:  # confidence threshold
                            xyxy = box.xyxy.cpu().numpy().reshape(-1)
                            x0, y0, x1, y1 = [int(v) for v in xyxy]
                            player_bboxes.append((x0, y0, x1-x0, y1-y0))
                except Exception:
                    continue
            
            logging.info("Detected %s players in base frame", len(player_bboxes))
            
            # Step 2: Find suitable reference frame
            reference_frame = None
            reference_time = None
            
            # Search nearby frames for a clear view
            search_times = [target_time - 15, target_time + 15]
            for search_time in search_times:
                if search_time < 0 or search_time * fps >= total_frames:
                    continue
                    
                frame_num = int(search_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, candidate_frame = cap.read()
                if not ret or candidate_frame is None:
                    continue
                
                # Run YOLO on candidate frame
                results = self.yolo_model.predict(source=candidate_frame, verbose=False)[0]
                candidate_bboxes = []
                for box in getattr(results, "boxes", []):
                    try:
                        cls_id = int(box.cls.item())
                        if cls_id == 0:  # person class
                            conf = float(box.conf.item())
                            if conf > 0.5:
                                xyxy = box.xyxy.cpu().numpy().reshape(-1)
                                x0, y0, x1, y1 = [int(v) for v in xyxy]
                                candidate_bboxes.append((x0, y0, x1-x0, y1-y0))
                    except Exception:
                        continue
                
                # Check if this frame has clear areas where base frame has players
                is_suitable = True
                for base_bbox in player_bboxes:
                    bx, by, bw, bh = base_bbox
                    base_center = (bx + bw//2, by + bh//2)
                    
                    # Check if any player in candidate frame overlaps significantly with base occlusion
                    for cand_bbox in candidate_bboxes:
                        cx, cy, cw, ch = cand_bbox
                        cand_center = (cx + cw//2, cy + ch//2)
                        
                        # Calculate overlap
                        overlap_x = max(0, min(bx + bw, cx + cw) - max(bx, cx))
                        overlap_y = max(0, min(by + bh, cy + ch) - max(by, cy))
                        overlap_area = overlap_x * overlap_y
                        base_area = bw * bh
                        
                        if overlap_area > 0.3 * base_area:  # 30% overlap threshold
                            is_suitable = False
                            break
                    if not is_suitable:
                        break
                
                if is_suitable:
                    reference_frame = candidate_frame
                    reference_time = search_time
                    logging.info("Found suitable reference frame at %ss", search_time)
                    break
            
            if reference_frame is None:
                logging.info("No suitable reference frame found, using base frame")
                return base_frame
            
            # Step 3: Align frames using homography
            logging.info("Aligning frames using homography...")
            
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=1000)
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(base_frame, None)
            kp2, des2 = orb.detectAndCompute(reference_frame, None)
            
            if des1 is not None and des2 is not None:
                # Match descriptors
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Keep only the best matches
                good_matches = matches[:min(100, len(matches))]
                
                if len(good_matches) >= 10:  # Need minimum matches for homography
                    # Get coordinates of good matches
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Calculate the Homography matrix
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        # Step 4: Warp and combine
                        h, w, c = base_frame.shape
                        warped_ref_frame = cv2.warpPerspective(reference_frame, M, (w, h))
                        
                        # Create occlusion mask from player bounding boxes
                        occlusion_mask = np.zeros(base_frame.shape[:2], dtype=np.uint8)
                        for bbox in player_bboxes:
                            x, y, w_bbox, h_bbox = bbox
                            cv2.rectangle(occlusion_mask, (x, y), (x+w_bbox, y+h_bbox), 255, -1)
                        
                        # Dilate mask slightly to ensure complete coverage
                        kernel = np.ones((5, 5), np.uint8)
                        occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=1)
                        
                        # Create the final clean frame
                        clean_frame = np.where(occlusion_mask[:, :, None] == 255, warped_ref_frame, base_frame)
                        logging.info("Successfully created clean frame using homography alignment")
                        return clean_frame.astype(np.uint8)
                    else:
                        logging.info("Homography calculation failed, using base frame")
                        return base_frame
                else:
                    logging.info("Insufficient feature matches, using base frame")
                    return base_frame
            else:
                logging.info("Feature detection failed, using base frame")
                return base_frame
                
        finally:
            cap.release()
    
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
            points = contour.reshape(-1, 2)
            
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
        
        Args:
            horizontal_lines: List of horizontal lines
            
        Returns:
            The baseline line or None if not found
        """
        baseline = None
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2-x1) >= self.MIN_BASELINE_LEN:  # long enough for baseline
                if baseline is None:
                    baseline = line
                elif (y1+y2)/2 > (baseline[0][1]+baseline[0][3])/2:  # mean y of current line is greater
                    baseline = line
        return baseline
    
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
    
    def process_video(self, video_path: str, target_time: int = 60) -> Tuple[Optional[np.ndarray], np.ndarray, dict]:
        """
        Main wrapper method to process a video and return the "out" mask.
        
        Args:
            video_path: Path to the video file
            target_time: Time in seconds to extract frame from (default: 60)
            
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
            print("Right doubles sideline: FOUND")
        else:
            print("Right doubles sideline: NOT FOUND")
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
            print("Left doubles sideline: FOUND")
        else:
            print("Left doubles sideline: NOT FOUND")
            failures.append("LEFT SIDELINE")
        
        # Draw extended doubles sidelines in pink and yellow (if both sidelines are found)
        if left_doubles_sideline is not None and right_doubles_sideline is not None:
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
            
            print(f"Extended doubles sidelines drawn (left: pink, right: yellow, shift: {dynamic_shift:.1f}px)")
        
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
        
        print(f"{side.capitalize()} side: Found {len(close_lines)} lines close to baseline out of {len(diagonal_lines)} total")
        
        # Decision tree based on number of close lines
        if len(close_lines) == 1:
            # Branch 1: Exactly ONE line is close to baseline
            print(f"{side.capitalize()} side: Branch 1 - One close line (assumed singles sideline)")
            
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
                print(f"{side.capitalize()} side: No valid outward candidate found")
                return None, False
                
        elif len(close_lines) == 2:
            # Branch 2: Exactly TWO lines are close to baseline
            print(f"{side.capitalize()} side: Branch 2 - Two close lines (singles and doubles sidelines)")
            
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
            print(f"{side.capitalize()} side: Branch 3 - Ambiguous case ({len(close_lines)} close lines)")
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
    
    # Process the video and get the mask
    out_mask, clean_frame, metadata = detector.process_video(video_path, target_time=60)
    
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
