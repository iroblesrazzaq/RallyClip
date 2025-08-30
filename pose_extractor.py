import cv2
import torch
import sys
import time
import os
import numpy as np
from ultralytics import YOLO


class PoseExtractor:
    """
    A class that uses YOLOv8-pose model to extract raw numerical pose data
    from video segments and saves it to compressed .npz files.
    """
    
    def __init__(self, model_path='yolov8n-pose.pt'):
        """
        Initialize the PoseExtractor with a YOLOv8-pose model.
        
        Args:
            model_path (str): Path to the YOLOv8-pose model file.
                             Defaults to 'yolov8s-pose.pt'
        """
        # Try to use MPS if available, fallback to CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(f"Using device: {self.device.upper()} (MPS available)")
        else:
            self.device = "cpu"
            print(f"Using device: {self.device.upper()} (MPS not available, using CPU)")
        
        self.model = YOLO(model_path)
        print(f"YOLOv8-pose model loaded successfully from: {model_path}")
    
    def extract_pose_data(self, video_path, start_time_seconds=0, duration_seconds=60, target_fps=15):
        """
        Extract raw pose data from a video segment and save to .npz file.
        
        Args:
            video_path (str): Path to the input video file
            start_time_seconds (int): Start time in seconds (default: 0)
            duration_seconds (int): Duration to process in seconds (default: 60)
            target_fps (int): Target frame rate for consistent temporal sampling (default: 15)
            
        Returns:
            str: Path to the created .npz file
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int(start_time_seconds * fps)
        end_frame = min(int(start_frame + (duration_seconds * fps)), total_frames)
        
        print(f"Processing frames {start_frame} to {end_frame} (Source FPS: {fps})")
        print(f"Target FPS: {target_fps}")
        
        # Calculate frame selection for target FPS
        frame_interval = fps / target_fps
        print(f"Frame interval: {frame_interval:.2f} frames")
        
        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize list to store all frame data
        all_frames_data = []
        total_frames_to_process = end_frame - start_frame
        
        print("Extracting pose data...")
        processed_frames = 0
        target_frame_count = 0
        
        for i in range(total_frames_to_process):
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error reading frame {i + start_frame}")
                break
            
            # Calculate target time for this frame
            current_time = start_time_seconds + (i / fps)
            target_time = target_frame_count / target_fps
            
            # Check if we should process this frame (within tolerance)
            if abs(current_time - target_time) <= (1 / fps) / 2:  # Within half a frame tolerance
                # Run YOLOv8-pose model on the frame (no plotting)
                results = self.model(frame, verbose=False, device=self.device, conf=0.05, imgsz=1920)
                
                # Extract raw numerical data
                frame_data = {}
                
                if len(results) > 0 and results[0].boxes is not None:
                    # Extract bounding boxes
                    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                    frame_data['boxes'] = boxes
                    
                    # Extract keypoints
                    keypoints = results[0].keypoints.xy.cpu().numpy()  # [num_persons, num_keypoints, 2]
                    frame_data['keypoints'] = keypoints
                    
                    # Extract keypoint confidences
                    keypoint_conf = results[0].keypoints.conf.cpu().numpy()  # [num_persons, num_keypoints]
                    frame_data['conf'] = keypoint_conf
                else:
                    # No detections in this frame
                    frame_data['boxes'] = np.array([])
                    frame_data['keypoints'] = np.array([])
                    frame_data['conf'] = np.array([])
                
                all_frames_data.append(frame_data)
                processed_frames += 1
                target_frame_count += 1
            else:
                # Skip this frame - add empty data to maintain frame alignment
                frame_data = {
                    'boxes': np.array([]),
                    'keypoints': np.array([]),
                    'conf': np.array([])
                }
                all_frames_data.append(frame_data)
            
            # Update progress bar
            progress = (i + 1) / total_frames_to_process * 100
            print(f"\rProgress: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}% (processed {processed_frames} frames)", end='', flush=True)
        
        print()  # New line after progress bar
        
        # Cleanup
        cap.release()
        
        print(f"✓ Successfully extracted pose data from {len(all_frames_data)} frames")
        
        # Save data to .npz file
        output_dir = "pose_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct descriptive filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_posedata_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s.npz"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save compressed data
        np.savez_compressed(output_path, frames=all_frames_data)
        print(f"✓ Pose data saved to: {output_path}")
        
        return output_path


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) >= 3:
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        video_path = sys.argv[4] if len(sys.argv) > 4 else "raw_videos/Monica Greene unedited tennis match play.mp4"
    else:
        start_time = 0
        duration = 10  # Default to 10 seconds for testing
        target_fps = 15
        video_path = "raw_videos/Monica Greene unedited tennis match play.mp4"
    
    # Start timing
    script_start_time = time.time()
    
    print("Initializing PoseExtractor...")
    pose_extractor = PoseExtractor()
    
    print(f"Processing video: {video_path}")
    print(f"Start time: {start_time}s, Duration: {duration}s, Target FPS: {target_fps}")
    
    # Extract pose data
    output_path = pose_extractor.extract_pose_data(
        video_path=video_path,
        start_time_seconds=start_time,
        duration_seconds=duration,
        target_fps=target_fps
    )
    
    if output_path is None:
        print("Failed to extract pose data. Exiting.")
        exit()
    
    # Calculate and print total runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
