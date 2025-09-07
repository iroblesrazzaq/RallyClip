#!/usr/bin/env python3
"""
Draw script for visualizing pose data on videos.

This script takes an NPZ file containing pose data and draws it on the corresponding video.
It can handle both raw and preprocessed NPZ files.

Usage:
    python draw.py --npz-path <path> [--start-time <seconds>] [--duration <seconds>]
    python draw.py --npz-dir <dir> [--start-time <seconds>] [--duration <seconds>] [--draw-all]
"""

import os
import sys
import argparse
import numpy as np
import cv2
import glob
from pathlib import Path

# COCO keypoint connections
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Colors for different elements
COLORS = {
    'bbox': (0, 255, 0),  # Green
    'keypoints': (0, 0, 255),  # Red
    'connections': (255, 0, 0),  # Blue
    'player1': (255, 0, 0),  # Red
    'player2': (0, 255, 0),  # Green
    'court_mask': (0, 0, 255, 128)  # Transparent red
}

def get_video_path_from_npz(npz_path):
    """Get the corresponding video path from the NPZ file path."""
    # Extract the video filename from the NPZ filename
    npz_filename = os.path.basename(npz_path)
    video_filename = npz_filename
    
    # Remove pose data suffix if present
    if "_posedata_" in video_filename:
        video_filename = video_filename.split("_posedata_")[0] + ".mp4"
    else:
        # Remove extension and add .mp4
        video_filename = os.path.splitext(video_filename)[0] + ".mp4"
    
    # Look in raw_videos directory
    raw_videos_dir = "raw_videos"
    video_path = os.path.join(raw_videos_dir, video_filename)
    
    # Try with .mov extension if .mp4 doesn't exist
    if not os.path.exists(video_path):
        video_path = os.path.join(raw_videos_dir, os.path.splitext(video_filename)[0] + ".mov")
    
    return video_path

def is_preprocessed_npz(npz_path):
    """Determine if the NPZ file is preprocessed or raw."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        # Preprocessed files have 'frames', 'targets', 'near_players', 'far_players'
        # Raw files have 'frames' with bounding boxes, keypoints, conf
        return 'targets' in data and 'near_players' in data and 'far_players' in data
    except Exception as e:
        print(f"Error reading NPZ file: {e}")
        return False

def draw_raw_pose_data(frame, frame_data):
    """Draw raw pose data on a frame."""
    if frame_data is None:
        return frame
    
    boxes = frame_data.get('boxes', np.array([]))
    keypoints = frame_data.get('keypoints', np.array([]))
    confs = frame_data.get('conf', np.array([]))
    
    # Draw each detection
    for i, box in enumerate(boxes):
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['bbox'], 2)
        
        # Draw keypoints
        if i < len(keypoints):
            kps = keypoints[i]
            for kp in kps:
                x, y = map(int, kp)
                cv2.circle(frame, (x, y), 3, COLORS['keypoints'], -1)
            
            # Draw connections
            for connection in COCO_CONNECTIONS:
                if connection[0] < len(kps) and connection[1] < len(kps):
                    pt1 = tuple(map(int, kps[connection[0]]))
                    pt2 = tuple(map(int, kps[connection[1]]))
                    cv2.line(frame, pt1, pt2, COLORS['connections'], 2)
        
        # Draw confidence if available
        if i < len(confs):
            conf_text = f"{np.mean(confs[i]):.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['bbox'], 1)
    
    return frame

def draw_player_pose(frame, player_data, color, label):
    """Draw a single player's pose on a frame."""
    if player_data is None:
        return frame
    
    # Draw bounding box
    box = player_data.get('box', None)
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw keypoints
    kps = player_data.get('keypoints', None)
    if kps is not None:
        for kp in kps:
            x, y = map(int, kp)
            cv2.circle(frame, (x, y), 3, color, -1)
        
        # Draw connections
        for connection in COCO_CONNECTIONS:
            if connection[0] < len(kps) and connection[1] < len(kps):
                pt1 = tuple(map(int, kps[connection[0]]))
                pt2 = tuple(map(int, kps[connection[1]]))
                cv2.line(frame, pt1, pt2, color, 2)
    
    return frame

def draw_preprocessed_pose_data(frame, frame_idx, frames_data, near_players, far_players, court_mask=None):
    """Draw preprocessed pose data on a frame."""
    # Draw court mask if available
    if court_mask is not None:
        # Create a transparent overlay
        overlay = frame.copy()
        # Make areas outside the court red (where mask == 255)
        mask_255 = (court_mask == 255)
        overlay[mask_255] = [0, 0, 255]  # Red
        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw near player (Player 1)
    if frame_idx < len(near_players):
        near_player = near_players[frame_idx]
        if near_player is not None:
            frame = draw_player_pose(frame, near_player, COLORS['player1'], "Player 1")
    
    # Draw far player (Player 2)
    if frame_idx < len(far_players):
        far_player = far_players[frame_idx]
        if far_player is not None:
            frame = draw_player_pose(frame, far_player, COLORS['player2'], "Player 2")
    
    return frame

def draw_npz_on_video(npz_path, start_time=0, duration=99999):
    """Draw pose data from NPZ file on corresponding video."""
    print(f"Processing: {npz_path}")
    
    # Get corresponding video path
    video_path = get_video_path_from_npz(npz_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Determine if this is a preprocessed or raw NPZ file
    is_preprocessed = is_preprocessed_npz(npz_path)
    print(f"  File type: {'Preprocessed' if is_preprocessed else 'Raw'}")
    
    # Load pose data
    try:
        data = np.load(npz_path, allow_pickle=True)
        if is_preprocessed:
            frames_data = data['frames']
            targets = data['targets']
            near_players = data['near_players']
            far_players = data['far_players']
            court_mask = data['court_mask'] if 'court_mask' in data else None
        else:
            frames_data = data['frames']
            targets = None
            near_players = None
            far_players = None
            court_mask = None
    except Exception as e:
        print(f"  ❌ Error loading NPZ file: {e}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Error opening video file: {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate start and end frames
    start_frame = int(start_time * fps)
    end_frame = min(int((start_time + duration) * fps), total_frames)
    
    print(f"  Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"  Processing frames: {start_frame} to {end_frame}")
    
    # Determine output path
    npz_dir = os.path.dirname(npz_path)
    if "preprocessed" in npz_dir:
        output_base_dir = "sanity_check_clips/preprocessed"
        # Extract the parameter subdir
        rel_path = os.path.relpath(npz_dir, "pose_data/preprocessed")
    else:
        output_base_dir = "sanity_check_clips/raw"
        # Extract the parameter subdir
        rel_path = os.path.relpath(npz_dir, "pose_data/raw")
    
    output_dir = os.path.join(output_base_dir, rel_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    npz_filename = os.path.basename(npz_path)
    output_filename = f"draw_{os.path.splitext(npz_filename)[0]}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"  Output path: {output_path}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"  ❌ Error creating output video file: {output_path}")
        cap.release()
        return False
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    frame_count = 0
    current_frame = start_frame
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw pose data on frame
        if is_preprocessed:
            if current_frame < len(frames_data):
                frame = draw_preprocessed_pose_data(
                    frame, current_frame, frames_data, near_players, far_players, court_mask
                )
        else:
            if current_frame < len(frames_data):
                frame = draw_raw_pose_data(frame, frames_data[current_frame])
        
        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        current_frame += 1
        
        # Progress indicator
        if frame_count % 30 == 0:
            print(f"    Processed {frame_count} frames")
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"  ✅ Finished processing {frame_count} frames")
    print(f"  Output saved to: {output_path}")
    return True

def draw_all_npz_in_dir(npz_dir, start_time=0, duration=99999):
    """Draw all NPZ files in a directory."""
    print(f"Processing all NPZ files in: {npz_dir}")
    
    # Find all NPZ files in the directory
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    
    if not npz_files:
        print(f"  ❌ No NPZ files found in {npz_dir}")
        return False
    
    print(f"  Found {len(npz_files)} NPZ files")
    
    successful = 0
    failed = 0
    
    for npz_file in npz_files:
        try:
            if draw_npz_on_video(npz_file, start_time, duration):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Failed to process {npz_file}: {e}")
            failed += 1
    
    print(f"  Summary: {successful} successful, {failed} failed")
    return failed == 0

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Draw pose data on videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python draw.py --npz-path pose_data/raw/s_0.05_0_60_15/video.npz
    python draw.py --npz-path pose_data/preprocessed/s_0.05_0_60_15/video_preprocessed.npz --start-time 10 --duration 30
    python draw.py --npz-dir pose_data/raw/s_0.05_0_60_15 --draw-all
        """
    )
    
    parser.add_argument('--npz-path', type=str, help='Path to NPZ file')
    parser.add_argument('--npz-dir', type=str, help='Path to directory containing NPZ files')
    parser.add_argument('--start-time', type=float, default=0, help='Start time in seconds (default: 0)')
    parser.add_argument('--duration', type=float, default=99999, help='Duration in seconds (default: 99999)')
    parser.add_argument('--draw-all', action='store_true', help='Draw all NPZ files in directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.npz_path and not args.npz_dir:
        print("❌ Error: Either --npz-path or --npz-dir must be specified")
        return 1
    
    if args.npz_path and args.npz_dir:
        print("❌ Error: Cannot specify both --npz-path and --npz-dir")
        return 1
    
    if args.npz_path:
        # Process single NPZ file
        try:
            if draw_npz_on_video(args.npz_path, args.start_time, args.duration):
                print("\n🎉 Successfully processed NPZ file!")
                return 0
            else:
                print("\n💥 Failed to process NPZ file!")
                return 1
        except Exception as e:
            print(f"\n💥 Error processing NPZ file: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.npz_dir:
        # Process all NPZ files in directory
        try:
            if draw_all_npz_in_dir(args.npz_dir, args.start_time, args.duration):
                print("\n🎉 Successfully processed all NPZ files!")
                return 0
            else:
                print("\n💥 Some NPZ files failed to process!")
                return 1
        except Exception as e:
            print(f"\n💥 Error processing NPZ files: {e}")
            import traceback
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    sys.exit(main())