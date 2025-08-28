
# %%
import sys
import math
import cv2
import numpy as np
import os
import math

# Import YOLO directly
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install ultralytics and ensure yolov8n.pt exists.")

# %%

MIN_BASELINE_LEN = 500
def main():
    # cell 1: get image 1 minute in - in prod have to make sure nothing is covering doubles lines at that point (players)

    vid_paths = ['raw_videos/Aditi Narayan ｜ Matchplay.mp4', 'raw_videos/Monica Greene unedited tennis match play.mp4', 
                'raw_videos/Anna Fijalkowska UNCUT MATCH PLAY (vs Felix Hein).mp4',
                'raw_videos/Otto Friedlein - unedited matchplay.mp4']
    vid_paths1 = [f"./raw_videos/{filename}" for filename in os.listdir('raw_videos') if 'mp4' in filename]
        # %%
    video_path = 'raw_videos/Monica Greene unedited tennis match play.mp4'
    video_path = vid_paths[0]
    for i, video_path in enumerate(vid_paths1):
        #if video_path != './raw_videos/Ryan Parkins - Unedited matchplay.mp4':
        #    continue
        #if video_path !='./raw_videos/9⧸5⧸15 Singles Uncut.mp4':
        #    continue

        #if video_path != './raw_videos/Unedited Points - Fall 2021 - Erkin Tootoonchi Moghaddam.mp4':
        #    continue

        print(video_path.split('/')[-1])
        # %%
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # Robust Background Reconstruction using YOLO and Homography
        if not YOLO_AVAILABLE:
            print("YOLO not available, falling back to single frame at 1 minute mark")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 60))
            ret, src = cap.read()
            if not ret or src is None:
                cap.release()
                raise RuntimeError("Could not read frame at 1 minute mark.")
        else:
            print("Using YOLO + Homography for robust background reconstruction")
            
            # Load YOLO model directly
            try:
                yolo_model = YOLO('yolov8n.pt')
                print("YOLO model loaded successfully")
            except Exception as e:
                print(f"YOLO model failed to load: {e}, falling back to single frame")
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 60))
                ret, src = cap.read()
                if not ret or src is None:
                    cap.release()
                    raise RuntimeError("Could not read frame at 1 minute mark.")
                else:
                    # Continue with the rest of the processing
                    pass
            else:
                # Step 1: Select base frame and find occlusions
                base_time = 60  # 60 seconds
                base_frame_num = int(base_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame_num)
                ret, base_frame = cap.read()
                if not ret or base_frame is None:
                    cap.release()
                    raise RuntimeError("Could not read base frame at 60 seconds.")
                
                # Run YOLO on base frame to detect players
                results = yolo_model.predict(source=base_frame, verbose=False)[0]
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
                
                # Search nearby frames for a clear view (15s increments for better player movement)
                search_times = [45, 75]  # 15 seconds before and after base frame
                for search_time in search_times:
                    frame_num = int(search_time * fps)
                    if frame_num >= total_frames:
                        continue
                        
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, candidate_frame = cap.read()
                    if not ret or candidate_frame is None:
                        continue
                    
                    # Run YOLO on candidate frame
                    results = yolo_model.predict(source=candidate_frame, verbose=False)[0]
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
                    src = base_frame
                else:
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
                                # Where mask is white (255), use warped_ref_frame; otherwise use base_frame
                                clean_frame = np.where(occlusion_mask[:, :, None] == 255, warped_ref_frame, base_frame)
                                src = clean_frame.astype(np.uint8)
                                
                                print("Successfully created clean frame using homography alignment")
                            else:
                                print("Homography calculation failed, using base frame")
                                src = base_frame
                        else:
                            print("Insufficient feature matches, using base frame")
                            src = base_frame
                    else:
                        print("Feature detection failed, using base frame")
                        src = base_frame
                
                        # No intermediate displays - proceed directly to line detection
        # --- 1. PRE-PROCESSING FOR EDGE DETECTION ---
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # --- 2. GENERATE THE TWO INITIAL MASKS ---
        # a) Find all high-contrast edges with Canny
        canny_edges = cv2.Canny(blurred, 50, 150)

        # b) Find all white pixels using LAB color space (the "loose" mask)
        lab_image = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        lower_white = np.array([145, 105, 105])
        upper_white = np.array([255, 150, 150])
        white_mask = cv2.inRange(lab_image, lower_white, upper_white)

        # --- 3. CREATE THE "PROXIMITY MASK" FROM CANNY EDGES ---
        # Dilate the Canny edges to create a zone around them.
        # A larger kernel size increases the "distance" threshold.
        kernel = np.ones((4,4), np.uint8)
        dilated_edges_mask = cv2.dilate(canny_edges, kernel, iterations=1)

        # --- 4. FIND THE INTERSECTION TO GET THE REFINED MASK ---
        # Keep only the pixels that are in BOTH the white mask AND the dilated edge mask
        refined_lines_mask = cv2.bitwise_and(white_mask, dilated_edges_mask)

        # (Optional) Clean up the final mask by closing small gaps
        refined_lines_mask = cv2.morphologyEx(refined_lines_mask, cv2.MORPH_CLOSE, kernel)

        # --- 5. VISUALIZE THE RESULTS ---
        final_result = cv2.bitwise_and(src, src, mask=refined_lines_mask)

        # No intermediate displays - proceed to ROI masking
        
        height, width = src.shape[:2]

        # --- Create a single-channel (grayscale) mask ---
        # Masks must be single-channel arrays of type uint8.
        roi_mask = np.zeros((height, width), dtype=np.uint8)

        # Define the cutoff points for the top and corners
        top_cutoff = int(height * 0.35)
        corner_cut_width = int(width * 0.15)
        corner_cut_height = int(0.5 * corner_cut_width)

        # Define the vertices of the polygon you want to KEEP
        roi_vertices = np.array([
            (0, height),                                  # Bottom-left
            (0, top_cutoff + corner_cut_height),          # Left edge, below the corner cut
            (corner_cut_width, top_cutoff),               # Top edge, after the left corner cut
            (width - corner_cut_width, top_cutoff),       # Top edge, before the right corner cut
            (width, top_cutoff + corner_cut_height),      # Right edge, below the corner cut
            (width, height)                               # Bottom-right
        ], dtype=np.int32)
        # trying without corner mask
        roi_vertices = np.array([
            (0, height),                                  # Bottom-left
            (0, top_cutoff),          # Left edge, below the corner cut
            (width, top_cutoff),      # Right edge, below the corner cut
            (width, height)                               # Bottom-right
        ], dtype=np.int32)


        # Fill the polygon area on the single-channel mask with white (255)
        cv2.fillPoly(roi_mask, [roi_vertices], 255)

        # Apply this ROI mask to your final color result
        masked_result = cv2.bitwise_and(refined_lines_mask, refined_lines_mask, mask=roi_mask)


        # No intermediate displays - proceed to line detection

                
        linesP = cv2.HoughLinesP(masked_result, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=40)

        # --- Draw the ORIGINAL, UNMERGED lines for comparison ---
        unmerged_lines_image = src.copy()
        colored_lines_image = src.copy()

        if linesP is None:
            raise TypeError('linesP is none, no hough lines detected')
    
        # Define font settings for displaying the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # BGR: White for high contrast
        font_thickness = 1
        
        # Get the horizontal center of the screen once
        screen_center_x = src.shape[1] / 2
        horiz, vert, sl_r_diag, sr_l_diag = [],  [],  [],  []



            





        for line in linesP:
            x1, y1, x2, y2 = line[0]
            
            # --- NEW: Write the endpoint coordinates onto the image ---
            coord_font_scale = 0.4 # Use a slightly smaller font for coordinates


            _, angle_deg = get_polar_angle(line[0])
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Normalize the angle to a [0, 180) degree range
            normalized_angle = int(angle_deg % 180)

            # Determine if the line is on the Left or Right side
            side_label = "L" if mid_x < screen_center_x else "R"

            # Default color is gray
            color = (128, 128, 182)  # BGR: Gray

            # Updated classification using normalized angle AND side label
            if normalized_angle < 15 or normalized_angle > 165: # Horizontal
                color = (0, 255, 0)        # BGR: Green
                horiz.append(line) # No need to store angle_deg for this version

            elif 75 < normalized_angle < 105: # Vertical
                color = (0, 0, 0)          # BGR: Black
                vert.append(line)

            elif 15 <= normalized_angle <= 75: # Positive Slope
                if side_label == "R":
                    color = (255, 0, 0)    # BGR: Blue
                    sr_l_diag.append(line)

            elif 105 <= normalized_angle <= 165: # Negative Slope
                if side_label == "L":
                    color = (0, 0, 255)    # BGR: Red
                    sl_r_diag.append(line)
            
            # Draw the line with the determined color
            cv2.line(colored_lines_image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Display the angle AND the side label
            display_text = f"{normalized_angle}{side_label}" # e.g., "42L" or "131R"
            text_position = (int(mid_x) + 5, int(mid_y))
            
            cv2.putText(colored_lines_image, display_text, text_position, font, 
                        font_scale, font_color, font_thickness, cv2.LINE_AA)

        # No intermediate displays - proceed to line merging


  
        # --- 1. DEFINE THE VISUAL MERGING FUNCTION (MODIFIED TO RETURN MASKS) ---
        def merge_lines_visually(lines_to_merge, image_shape, kernel_size=(5,25), iterations=2, min_contour_area=50):
            """
            Merges line segments by drawing them on a mask, using morphology to connect them,
            and finding the best-fit line for the resulting contours.
            NOW ALSO RETURNS THE INTERMEDIATE MASKS FOR VISUALIZATION.
            """
            # Create a blank mask for the initial drawing
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            if not lines_to_merge:
                # Return empty results if no lines are passed in
                return [], mask, np.zeros_like(mask)

            for line in lines_to_merge:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

            # Use a morphological CLOSE operation to connect nearby segments
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
            
            # Return the final lines AND the two intermediate masks
            return final_merged_lines, mask, closed_mask

        # --- 2. SET TUNABLE PARAMETERS FOR EACH LINE TYPE ---
        horiz_kernel = (5, 30) 
        diag_kernel = (2, 2)
        
        # --- 3. CALL FUNCTIONS AND VISUALIZE INTERMEDIATE STEPS ---
        
        # Process Horizontal Lines
        final_horiz_lines, horiz_mask, horiz_closed = merge_lines_visually(horiz, src.shape, kernel_size=horiz_kernel, iterations=2)
       

        # Process Right-Side Diagonals
        final_sr_lines, sr_mask, sr_closed = merge_lines_visually(sr_l_diag, src.shape, kernel_size=diag_kernel, iterations=2)

        # Process Left-Side Diagonals
        final_sl_lines, sl_mask, sl_closed = merge_lines_visually(sl_r_diag, src.shape, kernel_size=diag_kernel, iterations=2)


        # --- 4. DRAW FINAL MERGED LINES TO THE IMAGE ---
        final_lines_image = src.copy()
        
        # Draw horizontal lines (Green)
        for line in final_horiz_lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2-x1) >= MIN_BASELINE_LEN:
                cv2.line(final_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw right-side diagonals (Blue)
        for line in final_sr_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(final_lines_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Draw left-side diagonals (Red)
        for line in final_sl_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(final_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # No intermediate displays - proceed to court line detection

        # --- COURT LINE DETECTION PIPELINE ---
        
        # Step 1: Gather necessary information
        baseline = find_baseline(final_horiz_lines)
        if baseline is None:
            print("No valid baseline found, skipping this video")
            continue
            
        right_line_count = len(final_sr_lines)
        left_line_count = len(final_sl_lines)
        
        print(f"Baseline found: width = {abs(baseline[0][2] - baseline[0][0])}px")
        print(f"Right-side diagonal lines: {right_line_count}")
        print(f"Left-side diagonal lines: {left_line_count}")
        
        # Step 2: Process each side independently using decision tree
        right_doubles_sideline = process_side_decision_tree(
            final_sr_lines, baseline, src.shape[1], "right")
        left_doubles_sideline = process_side_decision_tree(
            final_sl_lines, baseline, src.shape[1], "left")
        
        # Step 3: Draw final results
        final_result_image = src.copy()
        
        # Track what failed for display
        failures = []
        
        # Draw baseline
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
                right_y_at_xmax = int(right_slope * src.shape[1] + right_intercept)
                cv2.line(final_result_image, (0, right_y_at_x0), (src.shape[1], right_y_at_xmax), (255, 0, 0), 5)
            else:
                # Vertical line
                cv2.line(final_result_image, (rx1, 0), (rx1, src.shape[0]), (255, 0, 0), 5)
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
                left_y_at_xmax = int(left_slope * src.shape[1] + left_intercept)
                cv2.line(final_result_image, (0, left_y_at_x0), (src.shape[1], left_y_at_xmax), (0, 0, 255), 5)
            else:
                # Vertical line
                cv2.line(final_result_image, (lx1, 0), (lx1, src.shape[0]), (0, 0, 255), 5)
            print("Left doubles sideline: FOUND")
        else:
            print("Left doubles sideline: NOT FOUND")
            failures.append("LEFT SIDELINE")
        
        # Draw extended doubles sidelines in pink (if both sidelines are found)
        if left_doubles_sideline is not None and right_doubles_sideline is not None:
            # Calculate shifted sidelines for visualization
            BASE_HORIZONTAL_SHIFT = 100
            screen_width = src.shape[1]
            bx1, by1, bx2, by2 = baseline[0]
            baseline_width = abs(bx2 - bx1)
            scale_factor = baseline_width / screen_width
            dynamic_shift = BASE_HORIZONTAL_SHIFT * scale_factor
            
            # Shift lines for visualization
            def shift_line_for_viz(line, shift_amount):
                x1, y1, x2, y2 = line[0]
                vx = x2 - x1
                vy = y2 - y1
                perp_x = -vy
                perp_y = vx
                length = np.sqrt(perp_x**2 + perp_y**2)
                if length == 0:
                    return (x1, y1, x2, y2)
                unit_perp_x = perp_x / length
                unit_perp_y = perp_y / length
                shift_x = unit_perp_x * shift_amount
                shift_y = unit_perp_y * shift_amount
                new_x1 = int(x1 + shift_x)
                new_y1 = int(y1 + shift_y)
                new_x2 = int(x2 + shift_x)
                new_y2 = int(y2 + shift_y)
                return (new_x1, new_y1, new_x2, new_y2)
            
            # Draw extended sidelines in pink - extended through the whole image
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
                left_y_at_xmax = int(left_slope * src.shape[1] + left_shifted_intercept)
                cv2.line(final_result_image, (0, left_y_at_x0), (src.shape[1], left_y_at_xmax), (147, 20, 255), 3)  # Pink
            else:
                # Vertical line shifted horizontally (leftward for left sideline)
                left_shifted_x = lx1 - dynamic_shift
                cv2.line(final_result_image, (left_shifted_x, 0), (left_shifted_x, src.shape[0]), (147, 20, 255), 3)  # Pink
            
            if right_slope != float('inf'):
                # For right sideline, shift outward (away from center of image)
                # Right sideline should be shifted to the right (positive x direction)
                # Since left sideline uses subtraction, right sideline should use addition
                right_shifted_intercept = right_intercept + dynamic_shift / np.sqrt(1 + right_slope**2)
                
                right_y_at_x0 = int(right_slope * 0 + right_shifted_intercept)
                right_y_at_xmax = int(right_slope * src.shape[1] + right_shifted_intercept)
                cv2.line(final_result_image, (0, right_y_at_x0), (src.shape[1], right_y_at_xmax), (0, 255, 255), 3)  # Yellow
            else:
                # Vertical line shifted horizontally (rightward for right sideline)
                right_shifted_x = rx1 + dynamic_shift
                cv2.line(final_result_image, (right_shifted_x, 0), (right_shifted_x, src.shape[0]), (0, 255, 255), 3)  # Yellow
            
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
    
        # Step 4: Estimate Playable Court Area
        print("\n=== ESTIMATING PLAYABLE COURT AREA ===")
        print("DEBUG: About to estimate playable area...")
        playable_area_mask = estimate_playable_court_area(
            left_doubles_sideline, right_doubles_sideline, baseline, src.shape)
        print("DEBUG: Playable area estimation completed")
        
        print("DEBUG: About to start visualization...")
        
        # Show the final result with extended sidelines
        print(f"Showing results for: {video_path.split('/')[-1]}")
        print(f"Image shape: {final_result_image.shape}")
        print("DEBUG: About to show image...")
        
        # Create named window and show image
        window_name = f'Court Detection - {video_path.split("/")[-1]}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, final_result_image)
        
        print("DEBUG: Image should be displayed now")
        print("Press any key to continue to next video...")
        
        # Wait for key press and ensure window is visible
        while True:
            key = cv2.waitKey(100) & 0xFF  # Wait 100ms at a time
            if key != 255:  # Any key pressed
                break
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
        print("DEBUG: Window closed, moving to next video")
        
        # Always release the video capture object at the end of each iteration
        cap.release()
            




        


'''
no. that's only the case for when the entire baseline is visible. attached are 3 screenshots - 1 perfect case where its 
very easy - entire baseline is visible, sidelines are connecting or very close, and there are no red herring lines.
 The second screenshot is when the entire baseline is visible, but there are red herring lines that we have to filter out. 
 The third case is when the entire baseline is not visible, and there are red herring lines. 
 the last case is when the entire baseline is not entirely visible but there aren't red herring lines. L
 ook through these and think carefully. How can we design heuristics for each one? 
 Maybe one strategy is if for each diagonal if there are only 2 sidelines, then those are the correct ones and we take 
 the outer one as the doubles sideline. For entire baseline visible and red herrings, then we can do the diagonal with 
 the end point closest to the relevant end of the baseline as the doubles sideline. For the case with red herrings and 
 entire baseline not visible, its tougher.



'''











def process_full_width_baseline_case(diagonal_lines, baseline, image_width, side):
    """
    Process the case where baseline is full-width (>98.5%) and there are more than 2 diagonal lines.
    
    Args:
        diagonal_lines: List of diagonal lines for one side
        baseline: The baseline line
        image_width: Width of the image
        side: "left" or "right" to indicate which side of the court
    
    Returns:
        tuple: (selected_doubles_sideline, is_valid)
    """
    bx1, by1, bx2, by2 = baseline[0]
    baseline_y = (by1 + by2) / 2
    tolerance = 100  # pixels - lines are "close" if within this distance of baseline
    
    # Find lines that are vertically close to the baseline
    close_lines = []
    for line in diagonal_lines:
        x1, y1, x2, y2 = line[0]
        # Get the lowest point (highest y-coordinate) of the line
        lowest_y = max(y1, y2)
        # Check if this line is close to the baseline
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
        
        # Calculate reference x (x-coordinate of singles line at mid_y)
        slope_singles, y_intercept_singles = get_line_equation(singles_line)
        if slope_singles is None:  # vertical line
            x_reference = sx1
        else:
            x_reference = (mid_y - y_intercept_singles) / slope_singles
        
        # Find the best candidate from far lines
        best_candidate = None
        min_distance = float('inf')
        
        for line in far_lines:
            x1, y1, x2, y2 = line[0]
            slope_candidate, y_intercept_candidate = get_line_equation(line)
            
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
            # Use midpoint of second line instead
            mid_x = (x3 + x4) / 2
        
        # Calculate y-coordinates for both lines at shared mid_x
        slope1, y_intercept1 = get_line_equation(line1)
        slope2, y_intercept2 = get_line_equation(line2)
        
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


def get_polar_angle(line):
    """Calculate the polar angle of a line given its endpoints."""
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_rad, angle_deg

def find_baseline(horizontal_lines):
    """Find the baseline from horizontal lines"""
    baseline = None
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2-x1) >= MIN_BASELINE_LEN:  # long enough for baseline
            if baseline is None:
                baseline = line
            elif (y1+y2)/2 > (baseline[0][1]+baseline[0][3])/2:  # mean y of current line is greater than mean y of previous baseline
                baseline = line
    return baseline

def find_outer_line(lines):
    """Find the outer line (lower y-coordinate) from a list of lines"""
    if len(lines) < 2:
        return lines[0] if lines else None
    
    # Try to find a shared x-point to compare y-values
    line1, line2 = lines[0], lines[1]
    
    # Try midpoint of line1
    x1, y1, x2, y2 = line1[0]
    mid_x1 = (x1 + x2) / 2
    
    if is_x_in_line_domain(line2, mid_x1):
        y1_at_mid = get_y_at_x(line1, mid_x1)
        y2_at_mid = get_y_at_x(line2, mid_x1)
        if y1_at_mid is not None and y2_at_mid is not None:
            return line1 if y1_at_mid < y2_at_mid else line2
    
    # Try midpoint of line2
    x1, y1, x2, y2 = line2[0]
    mid_x2 = (x1 + x2) / 2
    
    if is_x_in_line_domain(line1, mid_x2):
        y1_at_mid = get_y_at_x(line1, mid_x2)
        y2_at_mid = get_y_at_x(line2, mid_x2)
        if y1_at_mid is not None and y2_at_mid is not None:
            return line1 if y1_at_mid < y2_at_mid else line2
    
    # If no shared x-point found, use the line with lower average y-coordinate
    avg_y1 = (line1[0][1] + line1[0][3]) / 2
    avg_y2 = (line2[0][1] + line2[0][3]) / 2
    return line1 if avg_y1 < avg_y2 else line2

def validate_sideline_candidate(candidate, baseline, image_width):
    """Check if candidate line is close enough to baseline"""
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

def process_side_decision_tree(diagonal_lines, baseline, image_width, side):
    """
    Main decision tree for processing one side of the court.
    Returns the doubles sideline for that side, or None if not found.
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
        doubles_sideline = find_outer_line(diagonal_lines)
        if validate_sideline_candidate(doubles_sideline, baseline, image_width):
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
            # Full-width baseline case - use sophisticated decision tree
            doubles_sideline, is_valid = process_full_width_baseline_case(
                diagonal_lines, baseline, image_width, side)
            if is_valid:
                return doubles_sideline
            else:
                print(f"{side.capitalize()} side: Full-width baseline case failed")
                return None
        else:
            # Partially visible baseline - use distance-based approach
            doubles_sideline = process_partial_baseline_case(
                diagonal_lines, baseline, image_width, side)
            if validate_sideline_candidate(doubles_sideline, baseline, image_width):
                return doubles_sideline
            else:
                print(f"{side.capitalize()} side: Partial baseline case failed")
                return None

def process_partial_baseline_case(diagonal_lines, baseline, image_width, side):
    """Process case where baseline is partially visible (≤98.5% width)"""
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

def get_line_equation(line):
    """Get slope and y-intercept for a line segment"""
    x1, y1, x2, y2 = line[0]
    if x2 - x1 == 0:  # vertical line
        return None, x1  # slope=None, x_intercept
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    return slope, y_intercept

def get_y_at_x(line, x):
    """Get y-coordinate of line at given x-coordinate"""
    slope, y_intercept = get_line_equation(line)
    if slope is None:  # vertical line
        return None
    return slope * x + y_intercept

def is_x_in_line_domain(line, x):
    """Check if x-coordinate is within the domain of the line segment"""
    x1, y1, x2, y2 = line[0]
    return min(x1, x2) <= x <= max(x1, x2)

def estimate_playable_court_area(left_doubles_sideline, right_doubles_sideline, baseline, image_shape):
    """
    Create a quadrilateral polygon that defines the playable area by shifting the detected doubles sidelines outwards.
    
    Args:
        left_doubles_sideline: The coordinates of the detected left sideline [(x1, y1, x2, y2)]
        right_doubles_sideline: The coordinates of the detected right sideline [(x1, y1, x2, y2)]
        baseline: The coordinates of the detected baseline [(x1, y1, x2, y2)]
        image_shape: Tuple of (height, width) of the video frame
    
    Returns:
        numpy.ndarray: Binary mask where white area represents the estimated playable court area
    """
    if left_doubles_sideline is None or right_doubles_sideline is None or baseline is None:
        print("Warning: Missing court lines, cannot estimate playable area")
        return np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Step 1: Define Parameters and Calculate Dynamic Shift
    BASE_HORIZONTAL_SHIFT = 100  # pixels - desired buffer when baseline fills entire screen
    screen_width = image_shape[1]
    
    # Calculate baseline width
    bx1, by1, bx2, by2 = baseline[0]
    baseline_width = abs(bx2 - bx1)
    
    # Calculate scaling factor and dynamic shift
    scale_factor = baseline_width / screen_width
    dynamic_shift = BASE_HORIZONTAL_SHIFT * scale_factor
    
    print(f"Baseline width: {baseline_width}px, Screen width: {screen_width}px")
    print(f"Scale factor: {scale_factor:.3f}, Dynamic shift: {dynamic_shift:.1f}px")
    
    # Step 2: Perpendicularly Shift Each Sideline
    def shift_line_perpendicular(line, shift_amount, outward_direction):
        """
        Shift a line perpendicular to its direction by the specified amount.
        
        Args:
            line: Line coordinates [(x1, y1, x2, y2)]
            shift_amount: Distance to shift in pixels
            outward_direction: 1 for outward shift, -1 for inward shift
        
        Returns:
            tuple: New line coordinates (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = line[0]
        
        # Calculate the line vector
        vx = x2 - x1
        vy = y2 - y1
        
        # Calculate perpendicular normal vector (outward from court)
        # For a vector (vx, vy), the perpendicular is (-vy, vx)
        # We need to ensure it points outward from the court
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
    
    # Shift both sidelines outward
    shifted_left_sideline = shift_line_perpendicular(left_doubles_sideline, dynamic_shift, 1)
    shifted_right_sideline = shift_line_perpendicular(right_doubles_sideline, dynamic_shift, 1)
    
    print(f"Original left sideline: {left_doubles_sideline[0]}")
    print(f"Shifted left sideline: {shifted_left_sideline}")
    print(f"Original right sideline: {right_doubles_sideline[0]}")
    print(f"Shifted right sideline: {shifted_right_sideline}")
    
    # Step 3: Define the Four Corners of the Playable Area
    # Extract endpoints from shifted sidelines
    left_x1, left_y1, left_x2, left_y2 = shifted_left_sideline
    right_x1, right_y1, right_x2, right_y2 = shifted_right_sideline
    
    # Determine which endpoint is bottom and which is top for each sideline
    # Bottom endpoint has higher y-coordinate (lower in image)
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
    
    # Step 4: Create and Output the Playable Area Mask
    # Create a blank single-channel image
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

def filter_players_by_playable_area(player_bboxes, playable_area_mask):
    """
    Filter player bounding boxes to only include those within the playable area.
    
    Args:
        player_bboxes: List of player bounding boxes [(x, y, w, h), ...]
        playable_area_mask: Binary mask where white area is playable
    
    Returns:
        list: Filtered list of player bounding boxes within playable area
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


if __name__ == "__main__":
    main()
