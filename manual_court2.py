
# %%
import sys
import math
import cv2
import numpy as np
import os
import math

# %%
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
        top_cutoff = int(height * 0.20)
        corner_cut_width = int(width * 0.20)
        corner_cut_height = int(0.8 * corner_cut_width)

        # Define the vertices of the polygon you want to KEEP
        roi_vertices = np.array([
            (0, height),                                  # Bottom-left
            (0, top_cutoff + corner_cut_height),          # Left edge, below the corner cut
            (corner_cut_width, top_cutoff),               # Top edge, after the left corner cut
            (width - corner_cut_width, top_cutoff),       # Top edge, before the right corner cut
            (width, top_cutoff + corner_cut_height),      # Right edge, below the corner cut
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

                
        # (Your existing code to get linesP...)
# (Your existing code to get linesP...)
        linesP = cv2.HoughLinesP(masked_result, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=40)

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


        def merge_lines_visually(lines_to_merge, image_shape, kernel_size=(5,25), iterations=2, min_contour_area=50):
            """
            Merges line segments by drawing them on a mask, using morphology to connect them,
            and finding the best-fit line for the resulting contours.
            """
            if not lines_to_merge:
                return []

            # Create a blank mask and draw the raw line segments on it
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
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
                # Filter out small contours that are likely noise
                if cv2.contourArea(contour) < min_contour_area:
                    continue
                
                # Find the two most distant points in the contour to define the final line
                max_dist = 0
                p1_final, p2_final = None, None
                
                # Reshape contour points for easier iteration
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

        # --- 2. SET TUNABLE PARAMETERS FOR EACH LINE TYPE ---
        # For Horizontal lines, use a wide, short kernel to connect horizontal gaps
        horiz_kernel = (5, 30) 
        # For Diagonals, a more square kernel might be better
        diag_kernel = (2, 2)
        
        # --- 3. CALL THE FUNCTION FOR EACH CATEGORY ---
        final_horiz_lines = merge_lines_visually(horiz, src.shape, kernel_size=horiz_kernel, iterations=1)
        final_sr_lines = merge_lines_visually(sr_l_diag, src.shape, kernel_size=diag_kernel, iterations=1)
        final_sl_lines = merge_lines_visually(sl_r_diag, src.shape, kernel_size=diag_kernel, iterations=1)

        # --- 4. DRAW FINAL MERGED LINES TO THE IMAGE ---
        final_lines_image = src.copy()
        
        # Draw horizontal lines (Green)
        for line in final_horiz_lines:
            x1, y1, x2, y2 = line[0]
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
