import cv2
import sys
import time
import os
import numpy as np


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
    
    def annotate_video(self, video_path, data_path, start_time_seconds=0, duration_seconds=60):
        """
        Create an annotated video from pose data and original video.
        
        Args:
            video_path (str): Path to the original video file
            data_path (str): Path to the .npz pose data file
            start_time_seconds (int): Start time in seconds (default: 0)
            duration_seconds (int): Duration to process in seconds (default: 60)
            
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
        
        # Create output directory
        output_dir = "sanity_check_clips"
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_annotated_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Drawing loop
        num_frames_to_process = min(len(pose_data), end_frame - start_frame)
        
        print("Creating annotated video...")
        for i in range(num_frames_to_process):
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error reading frame {i + start_frame}")
                break
            
            # Get pose data for this frame
            frame_pose_data = pose_data[i]
            
            # Draw annotations
            self._draw_annotations(frame, frame_pose_data)
            
            # Write frame to video
            out.write(frame)
            
            # Update progress bar
            progress = (i + 1) / num_frames_to_process * 100
            print(f"\rProgress: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress bar
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✓ Annotated video saved to: {output_path}")
        return output_path
    
    def _draw_annotations(self, frame, frame_pose_data):
        """
        Draw bounding boxes and keypoints on a frame.
        
        Args:
            frame: OpenCV frame to draw on
            frame_pose_data: Dictionary containing 'boxes', 'keypoints', and 'conf'
        """
        boxes = frame_pose_data['boxes']
        keypoints = frame_pose_data['keypoints']
        confidences = frame_pose_data['conf']
        
        # Draw bounding boxes and keypoints for each detected person
        for person_idx in range(len(boxes)):
            # Draw bounding box
            if len(boxes) > 0:
                box = boxes[person_idx]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw keypoints
            if len(keypoints) > 0 and len(confidences) > 0:
                person_keypoints = keypoints[person_idx]
                person_conf = confidences[person_idx]
                
                for kp_idx, (kp_x, kp_y) in enumerate(person_keypoints):
                    if person_conf[kp_idx] >= self.keypoint_threshold:
                        cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (0, 0, 255), -1)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) >= 3:
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        video_path = sys.argv[3] if len(sys.argv) > 3 else "raw_videos/Monica Greene unedited tennis match play.mp4"
    else:
        start_time = 0
        duration = 10  # Default to 10 seconds for testing
        video_path = "raw_videos/Monica Greene unedited tennis match play.mp4"
    
    # Start timing
    script_start_time = time.time()
    
    # Dynamically construct data path
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    data_filename = f"{base_name}_posedata_{start_time}s_to_{start_time + duration}s.npz"
    data_path = os.path.join("pose_data", data_filename)
    
    print("Initializing VideoAnnotator...")
    video_annotator = VideoAnnotator()
    
    print(f"Video path: {video_path}")
    print(f"Data path: {data_path}")
    print(f"Start time: {start_time}s, Duration: {duration}s")
    
    # Create annotated video
    output_path = video_annotator.annotate_video(
        video_path=video_path,
        data_path=data_path,
        start_time_seconds=start_time,
        duration_seconds=duration
    )
    
    if output_path is None:
        print("Failed to create annotated video. Exiting.")
        exit()
    
    # Calculate and print total runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
