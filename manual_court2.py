
# %%
import sys
import math
import cv2
import numpy as np
import os
import math

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
    for video_path in vid_paths1:
        #if video_path != './raw_videos/Ryan Parkins - Unedited matchplay.mp4':
        #    continue
        #if video_path !='./raw_videos/9⧸5⧸15 Singles Uncut.mp4':
        #    continue
        
        print(video_path.split('/')[-1])
        # %%
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_num = fps*60 # 1 minute after recording starts

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)

        ret, src = cap.read()
        if not ret or src is None:
            cap.release()
            raise RuntimeError("Could not read first frame for playable area detection.")

        # %%
        cv2.imshow('window', src)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
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

        cv2.imshow('Original "Loose" White Mask', white_mask)
        cv2.imshow('Dilated Canny Edges (Zone of Interest)', dilated_edges_mask)
        cv2.imshow('Refined Lines Mask (Final)', refined_lines_mask)


        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
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


        cv2.imshow("ROI Mask", roi_mask)
        cv2.imshow("masked result", masked_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

                
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

        # Display the final result
        cv2.imshow('Angle and Side Classification', colored_lines_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


  
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

        # Display the image with all final merged lines
        cv2.imshow('Final Visually Merged Lines', final_lines_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # before getting into complex heuristics, let's do the ideal case: we have 2 of each type of diagonal line and the baseline
        baseline = None
        for line in final_horiz_lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2-x1) >= MIN_BASELINE_LEN: # long enough for baseline
                if baseline is None:
                    baseline = line
                elif (y1+y2)/2 > (baseline[0][1]+baseline[0][3])/2: # mean y of current line is greater than mean y of previous baseline
                    baseline = line
        if baseline is None:
            continue
        else:
            
            # now we're gonna get the 2 outer diagonals:
            # the trick is that at the same shared x point, the outer line will be higher visually on screen = lower y value
            # how to get the shared x point? heuristic - midpoint of one line, look at the y value there vs the y value of the other line eval at x
            # if that midpoint x is not in domain of the other line, try the other direction
            # else, raise error, will handle this later
            
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
                if baseline_width_percentage <= 95:
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
            
            # Print line counts for debugging
            print(f"Right-side diagonal lines: {len(final_sr_lines)}")
            print(f"Left-side diagonal lines: {len(final_sl_lines)}")
            
            # Handle cases where we have exactly 2 diagonal lines on each side
            if len(final_sr_lines) == 2 and len(final_sl_lines) == 2:
                # Process right-side diagonals (ideal case)
                right_doubles_sideline = find_outer_line(final_sr_lines)
                
                # Process left-side diagonals (ideal case)
                left_doubles_sideline = find_outer_line(final_sl_lines)
                
                # Validate both sidelines are close enough to baseline
                right_valid = validate_sideline_candidate(right_doubles_sideline, baseline, src.shape[1])
                left_valid = validate_sideline_candidate(left_doubles_sideline, baseline, src.shape[1])
                
                print(f"Right sideline validation: {right_valid}")
                print(f"Left sideline validation: {left_valid}")
                
                if right_valid and left_valid:
                    # Draw the identified doubles sidelines
                    final_lines_image3 = src.copy()
                    
                    # Draw baseline
                    cv2.line(final_lines_image3, (baseline[0][0], baseline[0][1]), 
                             (baseline[0][2], baseline[0][3]), (0, 255, 0), 3)
                    
                    # Draw right doubles sideline
                    cv2.line(final_lines_image3, (right_doubles_sideline[0][0], right_doubles_sideline[0][1]), 
                             (right_doubles_sideline[0][2], right_doubles_sideline[0][3]), (255, 0, 0), 5)
                    
                    # Draw left doubles sideline
                    cv2.line(final_lines_image3, (left_doubles_sideline[0][0], left_doubles_sideline[0][1]), 
                             (left_doubles_sideline[0][2], left_doubles_sideline[0][3]), (0, 0, 255), 5)
                    
                    cv2.imshow('Identified Doubles Sidelines (Ideal Case)', final_lines_image3)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()



        


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


if __name__ == "__main__":
    main()
