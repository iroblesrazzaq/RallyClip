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
    'bbox': (0, 255, 0),      # Green
    'keypoints': (0, 0, 255),  # Red
    'connections': (255, 0, 0),  # Blue
    'player1': (255, 165, 0),  # Orange
    'player2': (128, 0, 128),  # Purple
    'point_indicator': (0, 255, 255),  # Cyan for point indicator
    'not_point_indicator': (0, 0, 255),  # Red for not in point
    'skipped_indicator': (128, 128, 128)  # Gray for skipped frames
}

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
            for x, y in kps:
                if x > 0 and y > 0:  # Only draw valid keypoints
                    cv2.circle(frame, (int(x), int(y)), 3, COLORS['keypoints'], -1)
            
            # Draw connections
            for start_idx, end_idx in COCO_CONNECTIONS:
                if start_idx < len(kps) and end_idx < len(kps):
                    start_point = kps[start_idx]
                    end_point = kps[end_idx]
                    if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
                        cv2.line(frame, 
                                (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])),
                                COLORS['connections'], 2)
        
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
        for x, y in kps:
            if x > 0 and y > 0:  # Only draw valid keypoints
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        # Draw connections
        for start_idx, end_idx in COCO_CONNECTIONS:
            if start_idx < len(kps) and end_idx < len(kps):
                start_point = kps[start_idx]
                end_point = kps[end_idx]
                if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
                    cv2.line(frame, 
                            (int(start_point[0]), int(start_point[1])),
                            (int(end_point[0]), int(end_point[1])),
                            color, 2)
    
    return frame

def draw_preprocessed_pose_data(frame, frame_idx, frames_data, near_players, far_players, court_mask):
    """Draw preprocessed pose data on a frame."""
    # Draw court mask if available
    if court_mask is not None:
        # Overlay court mask with transparency
        overlay = frame.copy()
        overlay[court_mask > 0] = [0, 255, 0]  # Green for court area
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

def get_video_path_from_npz(npz_path):
    """Get the corresponding video path from an NPZ file path."""
    # Extract base name and remove _posedata suffix
    base_name = os.path.basename(npz_path)
    if '_posedata_' in base_name:
        # Raw pose data filename: video_name_posedata_Xs_to_Ys_yoloZ.npz
        # Extract everything before "_posedata_" as the video name
        video_name = base_name.split('_posedata_')[0] + '.mp4'
    else:
        # Preprocessed data filename: video_name_preprocessed.npz
        video_name = base_name.replace('_preprocessed.npz', '.mp4')
    
    # Look in raw_videos directory
    video_path = os.path.join('raw_videos', video_name)
    if os.path.exists(video_path):
        return video_path
    
    # Try other extensions
    for ext in ['.mov', '.avi', '.mkv']:
        video_path = os.path.join('raw_videos', video_name.replace('.mp4', ext))
        if os.path.exists(video_path):
            return video_path
    
    return video_path  # Return the .mp4 version if not found

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
    
    # Open video file
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
    npz_name = os.path.splitext(os.path.basename(npz_path))[0]
    output_filename = f"{npz_name}_{start_time}s_to_{start_time + duration}s.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Create output video writer
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
                # Add point indicator for preprocessed data
                if targets is not None and current_frame < len(targets):
                    is_in_point = targets[current_frame] == 1
                    point_text = "IN POINT" if is_in_point else "NOT IN POINT"
                    point_color = COLORS['point_indicator'] if is_in_point else COLORS['not_point_indicator']
                    # Position: top-left for both indicators (they're not flickering)
                    position = (20, 40)
                    cv2.putText(frame, point_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, point_color, 2)
        else:
            if current_frame < len(frames_data):
                frame_data = frames_data[current_frame]
                frame = draw_raw_pose_data(frame, frame_data)
                # Add point indicator for raw data
                annotation_status = frame_data.get('annotation_status', 0)
                if annotation_status == -100:
                    # This is a skipped frame due to temporal downsampling
                    point_text = "SKIPPED FRAME"
                    point_color = COLORS['skipped_indicator']
                    position = (20, 40)  # Top-left
                    cv2.putText(frame, point_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, point_color, 2)
                else:
                    # This is either in point (1) or not in point (0)
                    is_in_point = annotation_status == 1
                    point_text = "IN POINT" if is_in_point else "NOT IN POINT"
                    point_color = COLORS['point_indicator'] if is_in_point else COLORS['not_point_indicator']
                    position = (20, 40)  # Top-left
                    cv2.putText(frame, point_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, point_color, 2)
        
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
            print(f"  ❌ Error processing {npz_file}: {e}")
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
    python draw.py --npz-path pose_data/raw/yolos_0.05conf_15fps_0s_to_60s/video_posedata_0s_to_60s_yolos.npz
    python draw.py --npz-dir pose_data/preprocessed/s_0.05_0_60_15 --start-time 10 --duration 30
    python draw.py --npz-dir pose_data/raw/yolos_0.05conf_15fps_0s_to_60s --draw-all
        """
    )
    
    parser.add_argument('--npz-path', type=str, help='Path to NPZ file')
    parser.add_argument('--npz-dir', type=str, help='Directory containing NPZ files')
    parser.add_argument('--start-time', type=int, default=0, help='Start time in seconds (default: 0)')
    parser.add_argument('--duration', type=int, default=99999, help='Duration in seconds (default: all)')
    parser.add_argument('--draw-all', action='store_true', help='Draw all NPZ files in directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.npz_path and not args.npz_dir:
        parser.print_help()
        return 1
    
    if args.npz_path and args.npz_dir:
        print("❌ Error: Cannot specify both --npz-path and --npz-dir")
        return 1
    
    # Process based on arguments
    if args.npz_path:
        return 0 if draw_npz_on_video(args.npz_path, args.start_time, args.duration) else 1
    elif args.npz_dir:
        if args.draw_all:
            return 0 if draw_all_npz_in_dir(args.npz_dir, args.start_time, args.duration) else 1
        else:
            # Draw first NPZ file in directory
            npz_files = glob.glob(os.path.join(args.npz_dir, "*.npz"))
            if not npz_files:
                print(f"❌ No NPZ files found in {args.npz_dir}")
                return 1
            return 0 if draw_npz_on_video(npz_files[0], args.start_time, args.duration) else 1

if __name__ == "__main__":
    sys.exit(main())