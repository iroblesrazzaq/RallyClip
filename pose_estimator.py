import cv2
from ultralytics import YOLO
import os
import torch
import sys
import time


class PoseEstimator:
    """
    A class that uses YOLOv8-pose model to process video segments and extract
    bounding boxes and keypoints for all detected persons.
    """
    
    def __init__(self, model_path='yolov8n-pose.pt', use_mps=True):
        """
        Initialize the PoseEstimator with a YOLOv8-pose model.
        
        Args:
            model_path (str): Path to the YOLOv8-pose model file.
            use_mps (bool): Flag to attempt to use Apple's MPS backend.
        """
        if use_mps and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            if use_mps: # Only print warning if the user intended to use MPS
                print("MPS not available. Falling back to CPU.")

        print(f"Using device: {self.device.upper()}")
        self.model = YOLO(model_path)
        print(f"YOLOv8-pose model loaded successfully from: {model_path}")
    
    def process_video_segment(self, video_path, start_time_seconds=0, duration_seconds=60):
        """
        Process a specific time segment of a video and return annotated frames.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return [], {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time_seconds * fps)
        end_frame = min(int(start_frame + (duration_seconds * fps)), total_frames)
        
        print(f"Processing frames {start_frame} to {end_frame} (FPS: {fps})")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        annotated_frames = []
        total_frames_to_process = end_frame - start_frame
        
        # --- PROFILING SETUP ---
        total_time_decode = 0
        total_time_inference = 0
        total_time_draw = 0
        
        print("Processing frames...")
        for i in range(total_frames_to_process):
            # --- Profile Decode Time ---
            t1 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {i + start_frame}")
                break
            t2 = time.perf_counter()
            total_time_decode += (t2 - t1)
            
            # --- Profile Inference Time ---
            # CRITICAL: Pass the device parameter here
            results = self.model(frame, verbose=False, conf=0.12, imgsz=1920, device=self.device)
            t3 = time.perf_counter()
            total_time_inference += (t3 - t2)
            
            # --- Profile Draw Time ---
            annotated_frame = results[0].plot()
            t4 = time.perf_counter()
            total_time_draw += (t4 - t3)

            annotated_frames.append(annotated_frame)
            
            progress = (i + 1) / total_frames_to_process * 100
            print(f"\rProgress: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
        
        print()
        cap.release()
        
        print(f"✓ Successfully processed {len(annotated_frames)} frames")
        
        # --- PROFILING RESULTS ---
        profiling_data = {
            "decode": total_time_decode,
            "inference": total_time_inference,
            "draw": total_time_draw,
            "count": len(annotated_frames)
        }
        
        return annotated_frames, profiling_data


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Set to False to force CPU and compare performance
    USE_MPS_IF_AVAILABLE = False 
    # Switch between 'yolov8n-pose.pt' (nano) and 'yolov8s-pose.pt' (small)
    MODEL_TO_USE = 'yolov8n-pose.pt' 

    if len(sys.argv) == 3:
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
    else:
        start_time = 0
        duration = 10 # Using a shorter duration for quicker tests

    script_start_time = time.time()
    
    video_path = "raw_videos/Monica Greene unedited tennis match play.mp4"
    
    print("Initializing PoseEstimator...")
    pose_estimator = PoseEstimator(model_path=MODEL_TO_USE, use_mps=USE_MPS_IF_AVAILABLE)
    
    print(f"Processing video: {video_path}")
    print(f"Start time: {start_time}s, Duration: {duration}s")
    
    annotated_frames, profiling_data = pose_estimator.process_video_segment(
        video_path=video_path,
        start_time_seconds=start_time,
        duration_seconds=duration
    )
    
    if not annotated_frames:
        print("No frames were processed. Exiting.")
        exit()
        
    # --- PRINT PROFILING RESULTS ---
    count = profiling_data["count"]
    if count > 0:
        total_decode = profiling_data["decode"]
        total_inference = profiling_data["inference"]
        total_draw = profiling_data["draw"]
        
        print("\n--- 📊 Profiling Results ---")
        print(f"Frames processed: {count}")
        print(f"Average time per frame:")
        print(f"  - 🖼️ Video Decode:      {total_decode / count * 1000:.2f} ms")
        print(f"  - 🧠 Model Inference:     {total_inference / count * 1000:.2f} ms")
        print(f"  - 🎨 Annotation Drawing:  {total_draw / count * 1000:.2f} ms")
        total_avg = (total_decode + total_inference + total_draw) / count
        print(f"  - ⏱️ TOTAL:               {total_avg * 1000:.2f} ms")
        print("---------------------------\n")

    print(f"Saving {len(annotated_frames)} annotated frames to video...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    output_dir = "sanity_check_clips"
    os.makedirs(output_dir, exist_ok=True)
    
    # Make filename more descriptive for testing
    model_name = os.path.splitext(MODEL_TO_USE)[0].replace('yolov8', '')
    device_name = pose_estimator.device
    output_filename = f"test_{model_name}_{device_name}_{start_time}s_to_{start_time + duration}s.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in annotated_frames:
        out.write(frame)
    
    out.release()
    
    print(f"✓ Video saved to: {output_path}")
    
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")