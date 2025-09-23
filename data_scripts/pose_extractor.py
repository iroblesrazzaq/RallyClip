import cv2
import torch
import sys
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import av


class PoseExtractor:
    """
    A class that uses YOLOv8-pose model to extract raw numerical pose data
    from video segments and saves it to compressed .npz files.
    """
    
    def __init__(self, model_dir='models', model_path='yolov8s-pose.pt'):
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
        
        self.model_path = model_path
        self.model = YOLO(os.path.join(model_dir, self.model_path))
        print(f"YOLOv8-pose model loaded successfully from: {model_path}")

    def frame_iterator_pyav(self, video_path):
        """
        A memory-efficient iterator that yields frames and their true timestamps as floats.
        This handles both Constant and Variable Frame Rate video correctly.
        """
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                time_base = stream.time_base
                for frame in container.decode(stream):
                    # Convert the frame's Presentation Timestamp (PTS) to seconds
                    timestamp_sec = float(frame.pts * time_base)
                    # Yield the frame as a NumPy array and its timestamp
                    yield frame.to_ndarray(format='bgr24'), timestamp_sec
        except Exception as e:
            print(f"\n[PyAV Error] Failed to open or decode video: {e}")
            return
    
    def extract_pose_data(self, video_path, confidence_threshold, start_time_seconds=0, duration_seconds=60, target_fps=15, annotations_csv=None, progress_callback=None):
        """
        Extract raw pose data from a video segment and save to .npz file.
        
        Args:
            video_path (str): Path to the input video file
            confidence_threshold (float): Confidence threshold for the model - no default
            start_time_seconds (int): Start time in seconds (default: 0)
            duration_seconds (int): Duration to process in seconds (default: 60)
            target_fps (int): Target frame rate for consistent temporal sampling (default: 15)
            annotations_csv (str): Path to CSV file with point annotations (optional)
            
        Returns:
            str: Path to the created .npz file
        """
        # Load annotations if provided
        annotations = None
        # Fix: Handle the case where "None" is passed as a string
        if annotations_csv and annotations_csv != "None" and os.path.exists(annotations_csv):
            try:
                annotations = pd.read_csv(annotations_csv)
                print(f"Loaded annotations from {annotations_csv}")
                # Show the column names for debugging
                print(f"Annotation columns: {list(annotations.columns)}")
            except Exception as e:
                print(f"Warning: Could not load annotations from {annotations_csv}: {e}")
        elif annotations_csv and annotations_csv != "None":
            print(f"Warning: Annotation file not found at {annotations_csv}")
        
        print(f"Processing video with VFR-safe timestamp scheduler...")
        
        # Get total frames from metadata for the progress bar
        try:
            with av.open(video_path) as container:
                total_frames = container.streams.video[0].frames
        except Exception:
            total_frames = 0 # Fallback if duration cannot be read

        # Initialize the list to store all frame data
        all_frames_data = []
        processed_frames_count = 0
        
        # Initialize the timestamp scheduler
        next_target_timestamp = start_time_seconds
        
        # --- Start of Optimized Annotation Logic ---
        EPS = 1e-6  # small tolerance for float comparisons at boundaries
        if annotations is not None:
            starts = annotations['start_time'].to_numpy(dtype=float)
            ends   = annotations['end_time'].to_numpy(dtype=float)
            annotation_index = 0
            num_annotations = starts.size
        else:
            starts = ends = None
            annotation_index = 0
            num_annotations = 0
        # --- End of Optimized Annotation Logic ---
        
        # Create the robust PyAV frame iterator
        frame_generator = self.frame_iterator_pyav(video_path)

        # Main processing loop with a progress bar
        pbar = tqdm(frame_generator, total=total_frames, desc="Processing frames", unit="frame")
        last_report_ts = -1.0  # throttle duplicate reports on same second
        for frame, current_timestamp in pbar:
            
            # Skip frames before our desired start time
            if current_timestamp < start_time_seconds:
                continue
            
            # Stop processing if we have exceeded the desired duration
            if current_timestamp > (start_time_seconds + duration_seconds):
                break

            # --- The Core Scheduling Logic ---
            # Check if the current frame's time has met or passed our scheduled target time
            if current_timestamp >= next_target_timestamp:
                
                # This frame is SELECTED for processing
                processed_frames_count += 1
                
                # --- [EXISTING LOGIC] ---
                # Run YOLOv8-pose model on the frame
                results = self.model(frame, verbose=False, device=self.device, conf=confidence_threshold, imgsz=1920)
                
                # --- [EXISTING LOGIC] ---
                # Extract raw numerical data (boxes, keypoints, confidences)
                frame_data = {}
                if len(results) > 0 and results[0].boxes is not None:
                    frame_data['boxes'] = results[0].boxes.xyxy.cpu().numpy()
                    frame_data['keypoints'] = results[0].keypoints.xy.cpu().numpy()
                    frame_data['conf'] = results[0].keypoints.conf.cpu().numpy()
                else:
                    frame_data['boxes'] = np.array([])
                    frame_data['keypoints'] = np.array([])
                    frame_data['conf'] = np.array([])
                
                # --- [NEW EFFICIENT LOGIC] ---
                annotation_status = 0
                if num_annotations > 0:
                    # Advance past intervals that ended before this timestamp
                    while (annotation_index < num_annotations - 1) and (current_timestamp > ends[annotation_index] + EPS):
                        annotation_index += 1
                    # Check if the timestamp lies within the current interval (inclusive with epsilon)
                    if (starts[annotation_index] - EPS) <= current_timestamp <= (ends[annotation_index] + EPS):
                        annotation_status = 1
                frame_data['annotation_status'] = annotation_status
                
                # --- CRITICAL STEP: Update the pacemaker for the *next* beat ---
                next_target_timestamp += (1.0 / target_fps)

            else:
                # --- This frame is SKIPPED ---
                # Create an empty placeholder to maintain the 1:1 frame mapping
                frame_data = {
                    'boxes': np.array([]),
                    'keypoints': np.array([]),
                    'conf': np.array([]),
                    'annotation_status': -100  # Flag for skipped frames
                }

            # Add the result (either processed or skipped) to our final list
            all_frames_data.append(frame_data)
            
            # Update the progress bar's postfix to show how many frames were processed
            pbar.set_postfix({"Processed": processed_frames_count})

            # Report progress based on wall-clock timeline within requested segment
            if progress_callback is not None and duration_seconds > 0:
                # Clamp progress between 0 and 1 using timestamps
                frac = (current_timestamp - start_time_seconds) / float(duration_seconds)
                if frac < 0:
                    frac = 0.0
                if frac > 1:
                    frac = 1.0
                # Throttle to once per second boundary change to reduce overhead
                current_sec_bucket = int((frac * duration_seconds) if duration_seconds > 0 else 0)
                if current_sec_bucket != last_report_ts:
                    last_report_ts = current_sec_bucket
                    try:
                        progress_callback(frac)
                    except Exception:
                        pass

        pbar.close()

        print(f"✓ Successfully iterated through video. Processed {processed_frames_count} frames.")
        
        # --- [EXISTING LOGIC] ---
        # Save data to .npz file
        # Extract model size from model path (e.g., "yolov8s-pose.pt" -> "s")
        if 'yolov8' in self.model_path:
            model_size = self.model_path.split('yolov8')[1].split('-')[0]
        else:
            model_size = 's'  # Default fallback
        
        # Create subdirectory with model size, confidence threshold, fps, and time range
        subdir_name = f"yolo{model_size}_{confidence_threshold}conf_{target_fps}fps_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s"
        output_dir = os.path.join("pose_data", "raw", subdir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct descriptive filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_posedata_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s_yolo{model_size}.npz"
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
        confidence_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
        video_path = sys.argv[5] if len(sys.argv) > 5 else "raw_videos/Monica Greene unedited tennis match play.mp4"
        model_size = sys.argv[6] if len(sys.argv) > 6 else "s"
        annotations_csv = sys.argv[7] if len(sys.argv) > 7 else None
    else:
        start_time = 0
        duration = 10  # Default to 10 seconds for testing
        target_fps = 15
        confidence_threshold = 0.05
        video_path = "raw_videos/Monica Greene unedited tennis match play.mp4"
        model_size = "s"
        annotations_csv = None
    
    # Start timing
    script_start_time = time.time()
    
    # Construct model path based on model size
    model_path = f"yolov8{model_size}-pose.pt"
    
    print("Initializing PoseExtractor...")
    pose_extractor = PoseExtractor(model_path=model_path)
    
    print(f"Processing video: {video_path}")
    print(f"Start time: {start_time}s, Duration: {duration}s, Target FPS: {target_fps}, Model: {model_path}")
    if annotations_csv:
        print(f"Annotations CSV: {annotations_csv}")
    
    # Extract pose data
    output_path = pose_extractor.extract_pose_data(
        video_path=video_path,
        start_time_seconds=start_time,
        duration_seconds=duration,
        target_fps=target_fps,
        confidence_threshold=confidence_threshold,
        annotations_csv=annotations_csv
    )
    
    if output_path is None:
        print("Failed to extract pose data. Exiting.")
        exit()
    
    # Calculate and print total runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")