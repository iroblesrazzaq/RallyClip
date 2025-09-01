import cv2
import sys
import time
import os
import numpy as np
import re


class VideoAnnotator:
    """
    A lightweight class that reads pose data from .npz files and creates
    annotated videos without requiring torch or ultralytics.
    """
    
    def __init__(self, keypoint_draw_threshold=0.5):
        """
        Initialize the VideoAnnotator.
        
        Args:
            keypoint_draw_threshold (float): Confidence threshold for drawing keypoints.
                                            Defaults to 0.5
        """
        self.keypoint_threshold = keypoint_draw_threshold
        print(f"Keypoint draw threshold: {self.keypoint_threshold}")
    
    def annotate_video(self, video_path, data_path, start_time_seconds=0, duration_seconds=60, overwrite=False):
        """
        Create an annotated video from pose data and original video.
        
        Args:
            video_path (str): Path to the original video file
            data_path (str): Path to the .npz pose data file
            start_time_seconds (int): Start time in seconds (default: 0)
            duration_seconds (int): Duration to process in seconds (default: 60)
            overwrite (bool): Overwrite existing video file (default: False)
            
        Returns:
            str: Path to the created annotated video file
        """
        # Load pose data
        try:
            pose_data = np.load(data_path, allow_pickle=True)['frames']
            print(f"✓ Loaded pose data from: {data_path}")
            print(f"  - Number of frames: {len(pose_data)}")
        except FileNotFoundError:
            print(f"Error: Could not find pose data file: {data_path}")
            return None
        except Exception as e:
            print(f"Error loading pose data: {e}")
            return None
        
        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int(start_time_seconds * fps)
        end_frame = min(int(start_frame + (duration_seconds * fps)), total_frames)
        
        print(f"Video properties: {width}x{height}, {fps} FPS")
        print(f"Processing frames {start_frame} to {end_frame}")
        
        # Extract model size and confidence threshold from data path
        data_filename = os.path.basename(data_path)
        if '_yolo' in data_filename:
            model_size = data_filename.split('_yolo')[1].split('.')[0]
        else:
            model_size = 's'  # Default if not found
        
        # Extract confidence threshold, fps, and time range from data path (assuming it's in the subdirectory name)
        data_dir = os.path.dirname(data_path)
        if os.path.basename(data_dir).endswith('s'):
            # Extract confidence, fps, and time range from subdirectory name like "yolom_0.05conf_15fps_0s_to_60s"
            conf_match = re.search(r'_(\d+\.\d+)conf_(\d+)fps_(\d+)s_to_(\d+)s$', os.path.basename(data_dir))
            if conf_match:
                confidence_threshold = conf_match.group(1)
                fps = int(conf_match.group(2))  # Keep FPS as integer to match directory naming
                start_time = conf_match.group(3)
                end_time = conf_match.group(4)
            else:
                # Fallback to old format without fps
                conf_match = re.search(r'_(\d+\.\d+)conf_(\d+)s_to_(\d+)s$', os.path.basename(data_dir))
                if conf_match:
                    confidence_threshold = conf_match.group(1)
                    fps = 15.0  # Default fps
                    start_time = conf_match.group(2)
                    end_time = conf_match.group(3)
                else:
                    confidence_threshold = "0.05"
                    fps = 15.0  # Default fps
                    start_time = str(start_time_seconds)
                    end_time = str(start_time_seconds + duration_seconds)
        else:
            confidence_threshold = "0.05"  # Default
            fps = 15.0  # Default fps
            start_time = str(start_time_seconds)
            end_time = str(start_time_seconds + duration_seconds)
        
        # Check if this is filtered data (court_filtered_ prefix in directory name)
        data_dir_name = os.path.basename(data_dir)
        is_filtered = data_dir_name.startswith('court_filtered_')
        
        # Load court mask if this is filtered data
        court_mask = None
        if is_filtered:
            try:
                # Try to load the court mask for this video
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                mask_path = f"court_masks/{base_name}_court_mask.npz"
                
                if os.path.exists(mask_path):
                    mask_data = np.load(mask_path, allow_pickle=True)
                    court_mask = mask_data['mask']
                    print(f"✓ Loaded court mask: {court_mask.shape}")
                else:
                    print(f"⚠️  Court mask not found: {mask_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load court mask: {e}")
        
        # Create subdirectory with model size, confidence threshold, fps, and time range
        # Use integer FPS to match the directory naming convention from filtering
        fps_int = int(fps)
        subdir_name = f"yolo{model_size}_{confidence_threshold}conf_{fps_int}fps_{start_time}s_to_{end_time}s"
        
        # Choose output directory based on whether data is filtered
        if is_filtered:
            output_dir = os.path.join("sanity_check_clips", "court_filtered", subdir_name)
        else:
            output_dir = os.path.join("sanity_check_clips", "unfiltered", subdir_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_annotated_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s_yolo{model_size}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Check if output file already exists
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            if overwrite:
                print(f"🔄 Output file exists, overwriting: {output_path}")
            else:
                print(f"✓ Output file already exists, skipping: {output_path}")
                return output_path
        
        # Initialize video writer with original video FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
        
        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Calculate which frames to process for target FPS (same logic as pose extractor)
        total_frames_to_process = end_frame - start_frame
        target_frames = []
        for i in range(total_frames_to_process):
            current_time = start_time_seconds + (i / cap.get(cv2.CAP_PROP_FPS))
            target_frame_index = int(current_time * fps)
            target_time = target_frame_index / fps
            
            # Check if this frame should be processed (closest to target time)
            if abs(current_time - target_time) <= (1 / fps) / 2:
                target_frames.append(i)
        
        print(f"Will process {len(target_frames)} frames out of {total_frames_to_process} total frames")
        print(f"Creating annotated video at original {cap.get(cv2.CAP_PROP_FPS)} FPS...")
        
        # Drawing loop - write all frames, but only modify those with pose data
        
        for i in range(total_frames_to_process):
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error reading frame {i + start_frame}")
                break
            
            # Apply court mask overlay to all frames (if available)
            if court_mask is not None:
                self._draw_court_mask_overlay(frame, court_mask)
            
            # Check if this frame has pose data by calculating the target frame time
            current_time = start_time_seconds + (i / cap.get(cv2.CAP_PROP_FPS))
            target_frame_time = 0  # Will be updated for each pose data entry
            
            frame_has_pose_data = False
            frame_pose_data = None
            
            # Look for pose data that matches this frame time
            for pose_idx, pose_data_entry in enumerate(pose_data):
                # Calculate when this pose data should appear based on target FPS
                target_frame_time = pose_idx / fps
                
                # Check if current frame time matches target frame time (within tolerance)
                if abs(current_time - target_frame_time) <= (1 / fps) / 2:
                    frame_pose_data = pose_data_entry
                    frame_has_pose_data = True
                    break
            
            # Only draw annotations if this frame has pose data (non-empty)
            if frame_has_pose_data and frame_pose_data is not None:
                if len(frame_pose_data['boxes']) > 0 or len(frame_pose_data['keypoints']) > 0:
                    # Draw annotations
                    self._draw_annotations(frame, frame_pose_data)
            
            # Write frame to video (all frames, modified or not)
            out.write(frame)
            
            # Update progress bar
            progress = (i + 1) / total_frames_to_process * 100
            print(f"\rProgress: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress bar
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✓ Annotated video saved to: {output_path}")
        return output_path
    
    def _draw_annotations(self, frame, frame_pose_data):
        """
        Draw bounding boxes, keypoints, centroids, and court mask on a frame.
        
        Args:
            frame: OpenCV frame to draw on
            frame_pose_data: Dictionary containing 'boxes', 'keypoints', and 'conf'
        """
        boxes = frame_pose_data['boxes']
        keypoints = frame_pose_data['keypoints']
        confidences = frame_pose_data['conf']
        
        # Draw bounding boxes, keypoints, and centroids for each detected person
        for person_idx in range(len(boxes)):
            # Draw bounding box
            if len(boxes) > 0:
                box = boxes[person_idx]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate and draw centroid with pink dot
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 5, (147, 20, 255), -1)  # Pink dot
            
            # Draw keypoints
            if len(keypoints) > 0 and len(confidences) > 0:
                person_keypoints = keypoints[person_idx]
                person_conf = confidences[person_idx]
                
                for kp_idx, (kp_x, kp_y) in enumerate(person_keypoints):
                    if person_conf[kp_idx] >= self.keypoint_threshold:
                        cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (0, 0, 255), -1)
    
    def _draw_court_mask_overlay(self, frame, court_mask):
        """
        Draw the court mask overlay on the frame.
        
        Args:
            frame: OpenCV frame to draw on
            court_mask: Binary mask where white (255) represents areas outside the playable court
        """
        if court_mask is None:
            return
        
        # Create a transparent overlay for "out" areas
        # Convert frame to float for alpha blending
        frame_float = frame.astype(np.float32) / 255.0
        
        # Create a semi-transparent overlay for out areas
        # Use a subtle tint that doesn't completely block the view
        overlay = np.zeros_like(frame_float)
        
        # Apply a subtle red tint to "out" areas (very transparent)
        # This will make out areas slightly reddish but still visible
        mask_normalized = court_mask.astype(np.float32) / 255.0
        alpha = 0.3  # Very low alpha for transparency (0.0 = fully transparent, 1.0 = fully opaque)
        
        # Expand mask to 3 channels
        mask_3d = np.stack([mask_normalized] * 3, axis=2)
        
        # Apply subtle red tint to out areas while keeping them mostly transparent
        overlay[:, :, 2] = mask_normalized * alpha  # Add red channel
        
        # Blend: out areas get subtle red tint, in areas stay original
        blended = frame_float * (1 - mask_3d * alpha) + overlay
        
        # Convert back to uint8
        frame[:] = (blended * 255).astype(np.uint8)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) >= 3:
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        confidence_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
        video_path = sys.argv[5] if len(sys.argv) > 5 else "raw_videos/Monica Greene unedited tennis match play.mp4"
        model_size = sys.argv[6] if len(sys.argv) > 6 else "s"
        overwrite = sys.argv[7].lower() in ['true', '1', 'yes', 'y'] if len(sys.argv) > 7 else False
        # Check for optional --data-path parameter
        data_path_override = None
        for i, arg in enumerate(sys.argv):
            if arg == "--data-path" and i + 1 < len(sys.argv):
                data_path_override = sys.argv[i + 1]
                break
    else:
        start_time = 0
        duration = 10  # Default to 10 seconds for testing
        target_fps = 15
        confidence_threshold = 0.05
        video_path = "raw_videos/Monica Greene unedited tennis match play.mp4"
        model_size = "s"
        overwrite = False
        data_path_override = None
    
    # Start timing
    script_start_time = time.time()
    
    # Use provided data path or construct default path
    if data_path_override:
        data_path = data_path_override
    else:
        # Dynamically construct data path with model size, confidence threshold, fps, and time range
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        # Use integer FPS to match directory naming convention
        target_fps_int = int(target_fps)
        subdir_name = f"yolo{model_size}_{confidence_threshold}conf_{target_fps_int}fps_{start_time}s_to_{start_time + duration}s"
        data_filename = f"{base_name}_posedata_{start_time}s_to_{start_time + duration}s_yolo{model_size}.npz"
        data_path = os.path.join("pose_data", "unfiltered", subdir_name, data_filename)
    
    print("Initializing VideoAnnotator...")
    video_annotator = VideoAnnotator()
    
    print(f"Video path: {video_path}")
    print(f"Data path: {data_path}")
    print(f"Start time: {start_time}s, Duration: {duration}s, Target FPS: {target_fps}, Confidence: {confidence_threshold}")
    
    # Create annotated video
    output_path = video_annotator.annotate_video(
        video_path=video_path,
        data_path=data_path,
        start_time_seconds=start_time,
        duration_seconds=duration,
        overwrite=overwrite
    )
    
    if output_path is None:
        print("Failed to create annotated video. Exiting.")
        exit()
    
    # Calculate and print total runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
