import sys
import math
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Union

# Import YOLO directly
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install ultralytics and ensure yolov8n.pt exists.")


class CourtDetector:
    """
    A class for detecting tennis court boundaries and estimating playable areas from video frames.
    """
    
    def __init__(self, yolo_model_path: str = 'yolov8n.pt'):
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
                print("YOLO model loaded successfully")
            except Exception as e:
                print(f"YOLO model failed to load: {e}")
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
                print("YOLO not available, using single frame at target time")
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * target_time))
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError("Could not read frame at target time.")
                return frame
            
            # Use YOLO + Homography for robust background reconstruction
            print("Using YOLO + Homography for robust background reconstruction")
            
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
            
            print(f"Detected {len(player_bboxes)} players in base frame")
            
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
                    print(f"Found suitable reference frame at {search_time}s")
                    break
            
            if reference_frame is None:
                print("No suitable reference frame found, using base frame")
                return base_frame
            
            # Step 3: Align frames using homography
            print("Aligning frames using homography...")
            
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
                        print("Successfully created clean frame using homography alignment")
                        return clean_frame.astype(np.uint8)
                    else:
                        print("Homography calculation failed, using base frame")
                        return base_frame
                else:
                    print("Insufficient feature matches, using base frame")
                    return base_frame
            else:
                print("Feature detection failed, using base frame")
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
            print(f"{side.capitalize()} side: No diagonal lines found")
            return None
        elif line_count == 1:
            print(f"{side.capitalize()} side: Only one diagonal line found")
            return None
        elif line_count == 2:
            # Ideal case: exactly 2 lines
            print(f"{side.capitalize()} side: Ideal case (2 lines)")
            doubles_sideline = self._find_outer_line(diagonal_lines)
            if self._validate_sideline_candidate(doubles_sideline, baseline, image_width):
                return doubles_sideline
            else:
                print(f"{side.capitalize()} side: Validation failed for ideal case")
                return None
        else:
            # Red herring case: more than 2 lines
            print(f"{side.capitalize()} side: Red herring case ({line_count} lines)")
            
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
                    print(f"{side.capitalize()} side: Full-width baseline case failed")
                    return None
            else:
                # Partially visible baseline
                doubles_sideline = self._process_partial_baseline_case(
                    diagonal_lines, baseline, image_width, side)
                if self._validate_sideline_candidate(doubles_sideline, baseline, image_width):
                    return doubles_sideline
                else:
                    print(f"{side.capitalize()} side: Partial baseline case failed")
                    return None
    
    def estimate_playable_court_area(self, left_doubles_sideline: Optional[List], 
                                   right_doubles_sideline: Optional[List], 
                                   baseline: Optional[List], 
                                   image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a quadrilateral polygon that defines the playable area.
        
        Args:
            left_doubles_sideline: The left doubles sideline
            right_doubles_sideline: The right doubles sideline
            baseline: The baseline
            image_shape: Shape of the image (height, width)
            
        Returns:
            Binary mask where white area represents the estimated playable court area
        """
        if left_doubles_sideline is None or right_doubles_sideline is None or baseline is None:
            print("Warning: Missing court lines, cannot estimate playable area")
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Define Parameters and Calculate Dynamic Shift
        BASE_HORIZONTAL_SHIFT = 100  # pixels
        screen_width = image_shape[1]
        
        # Calculate baseline width
        bx1, by1, bx2, by2 = baseline[0]
        baseline_width = abs(bx2 - bx1)
        
        # Calculate scaling factor and dynamic shift
        scale_factor = baseline_width / screen_width
        dynamic_shift = BASE_HORIZONTAL_SHIFT * scale_factor
        
        print(f"Baseline width: {baseline_width}px, Screen width: {screen_width}px")
        print(f"Scale factor: {scale_factor:.3f}, Dynamic shift: {dynamic_shift:.1f}px")
        
        # Shift both sidelines outward
        shifted_left_sideline = self._shift_line_perpendicular(left_doubles_sideline, dynamic_shift, 1)
        shifted_right_sideline = self._shift_line_perpendicular(right_doubles_sideline, dynamic_shift, 1)
        
        # Define the four corners of the playable area
        left_x1, left_y1, left_x2, left_y2 = shifted_left_sideline
        right_x1, right_y1, right_x2, right_y2 = shifted_right_sideline
        
        # Determine which endpoint is bottom and which is top for each sideline
        if left_y1 > left_y2:
            left_bottom = (left_x1, left_y1)
            left_top = (left_x2, left_y2)
        else:
            left_bottom = (left_x2, left_y2)
            left_top = (left_x1, left_y1)
        
        if right_y1 > right_y2:
            right_bottom = (right_x1, right_y1)
            right_top = (right_x2, right_y2)
        else:
            right_bottom = (right_x2, right_y2)
            right_top = (right_x1, right_y1)
        
        # Define the four corners of the playable area
        corner_bottom_left = left_bottom
        corner_bottom_right = right_bottom
        corner_top_right = right_top
        corner_top_left = left_top
        
        print(f"Playable area corners:")
        print(f"  Bottom-left: {corner_bottom_left}")
        print(f"  Bottom-right: {corner_bottom_right}")
        print(f"  Top-right: {corner_top_right}")
        print(f"  Top-left: {corner_top_left}")
        
        # Create and output the playable area mask
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Define the polygon vertices
        vertices = np.array([
            corner_bottom_left,
            corner_bottom_right,
            corner_top_right,
            corner_top_left
        ], dtype=np.int32)
        
        # Fill the polygon with white (255)
        cv2.fillPoly(mask, [vertices], 255)
        
        return mask
    
    def process_video(self, video_path: str, target_time: int = 60) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Main wrapper method to process a video and return the playable court area.
        
        Args:
            video_path: Path to the video file
            target_time: Time in seconds to extract frame from (default: 60)
            
        Returns:
            Tuple of (playable_area_mask, processed_frame, metadata)
        """
        print(f"Processing video: {video_path}")
        
        # Step 1: Extract clean frame
        clean_frame = self.extract_clean_frame(video_path, target_time)
        
        # Step 2: Detect court lines
        horizontal_lines, vertical_lines, right_diagonals, left_diagonals = self.detect_court_lines(clean_frame)
        
        # Step 3: Merge lines
        merged_horizontal = self.merge_lines(horizontal_lines, clean_frame.shape, kernel_size=(5, 30))
        merged_right_diagonals = self.merge_lines(right_diagonals, clean_frame.shape, kernel_size=(2, 2))
        merged_left_diagonals = self.merge_lines(left_diagonals, clean_frame.shape, kernel_size=(2, 2))
        
        # Step 4: Find baseline
        baseline = self.find_baseline(merged_horizontal)
        if baseline is None:
            print("No valid baseline found")
            return np.zeros(clean_frame.shape[:2], dtype=np.uint8), clean_frame, {"error": "No baseline found"}
        
        # Step 5: Process sides
        right_doubles_sideline = self.process_side_decision_tree(
            merged_right_diagonals, baseline, clean_frame.shape[1], "right")
        left_doubles_sideline = self.process_side_decision_tree(
            merged_left_diagonals, baseline, clean_frame.shape[1], "left")
        
        # Step 6: Estimate playable area
        playable_area_mask = self.estimate_playable_court_area(
            left_doubles_sideline, right_doubles_sideline, baseline, clean_frame.shape)
        
        # Create metadata
        metadata = {
            "baseline_found": baseline is not None,
            "left_sideline_found": left_doubles_sideline is not None,
            "right_sideline_found": right_doubles_sideline is not None,
            "playable_area_pixels": np.sum(playable_area_mask > 0),
            "total_pixels": playable_area_mask.size,
            "playable_area_percentage": (np.sum(playable_area_mask > 0) / playable_area_mask.size) * 100
        }
        
        return playable_area_mask, clean_frame, metadata
    
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
    
    print(f"Filtered {len(player_bboxes)} players to {len(filtered_bboxes)} in playable area")
    return filtered_bboxes


# Example usage
if __name__ == "__main__":
    # Initialize the court detector
    detector = CourtDetector()
    
    # Example video path
    video_path = "raw_videos/example_video.mp4"
    
    # Process the video
    playable_area_mask, processed_frame, metadata = detector.process_video(video_path)
    
    print("Processing complete!")
    print(f"Metadata: {metadata}")
    
    # Save the playable area mask
    cv2.imwrite("playable_area_mask.png", playable_area_mask)
    print("Playable area mask saved as 'playable_area_mask.png'")
