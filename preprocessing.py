# For handling command-line arguments
import argparse

# For video processing
import cv2

# For data manipulation and reading CSV files
import pandas as pd

# For numerical operations, especially with arrays
import numpy as np

# For the AI models
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO

# It's also good practice to import os for handling file paths
import os


def load_models():
    """Load YOLO and MoveNet models."""
    print("Loading AI models...")
    
    # Load YOLO model
    yolo_model = YOLO('yolov8n.pt')  # 'n' is the nano version, fast and small
    
    # Load MoveNet model
    movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet_input_size = 192
    
    print("✅ Models loaded successfully.")
    return yolo_model, movenet_model, movenet_input_size


def run_movenet(input_image, movenet_model, movenet_input_size):
    """Runs MoveNet on a single image and returns keypoints."""
    # Resize and pad the image to the model's expected input size
    image_for_movenet = tf.image.resize_with_pad(
        tf.expand_dims(input_image, axis=0), movenet_input_size, movenet_input_size
    )
    # Run inference
    infer = movenet_model.signatures['serving_default']
    # Run inference
    outputs = infer(tf.cast(image_for_movenet, dtype=tf.int32))
    # Access the output tensor from the returned dictionary
    keypoints_with_scores = outputs['output_0']
    return keypoints_with_scores


def extract_raw_data(video_path, yolo_model, movenet_model, movenet_input_size):
    """Extract raw pose data from video using YOLO tracking for both players."""
    print(f"🏃‍♂️ Starting dual-player pose extraction from video: {video_path}")
    
    video = cv2.VideoCapture(video_path)
    all_frame_data = []
    frame_count = 0
    
    # Initialize player role assignments
    far_player_id = None
    near_player_id = None
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        # 1. Track players with YOLO (using tracking instead of detection)
        results = yolo_model.track(frame, persist=True, classes=[0], verbose=False) # class 0 is 'person'

        # 2. Assign player roles on first few frames
        if far_player_id is None and results[0].boxes is not None and results[0].boxes.id is not None:
            # Find players and assign roles based on position
            players = []
            for i, box in enumerate(results[0].boxes):
                if box.id is not None:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    track_id = int(box.id.cpu().numpy()[0])
                    players.append((track_id, y1))  # (id, y_position)
            
            if len(players) >= 2:
                # Sort by y position (lowest y = highest on screen = far player)
                players.sort(key=lambda x: x[1])
                far_player_id = players[0][0]  # Highest on screen (lowest y)
                near_player_id = players[1][0]  # Lowest on screen (highest y)
                print(f"🎾 Assigned player roles: Far player ID={far_player_id}, Near player ID={near_player_id}")
            elif len(players) == 1:
                # Only one player detected, assign as far player for now
                far_player_id = players[0][0]
                print(f"🎾 Only one player detected: Far player ID={far_player_id}")

        # 3. Extract poses for both players
        far_kps = None
        near_kps = None
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes):
                if box.id is not None:
                    track_id = int(box.id.cpu().numpy()[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Check if this is one of our assigned players
                    if track_id == far_player_id or track_id == near_player_id:
                        player_crop = frame[y1:y2, x1:x2]
                        
                        # Ensure crop is not empty
                        if player_crop.shape[0] > 0 and player_crop.shape[1] > 0:
                            keypoints_relative = run_movenet(player_crop, movenet_model, movenet_input_size)
                            
                            # Convert keypoints to absolute coordinates
                            kps = keypoints_relative[0, 0].numpy()
                            abs_kps = np.zeros_like(kps)
                            abs_kps[:, 0] = kps[:, 0] * (y2 - y1) + y1 # y-coordinate
                            abs_kps[:, 1] = kps[:, 1] * (x2 - x1) + x1 # x-coordinate
                            abs_kps[:, 2] = kps[:, 2] # score
                            
                            # Assign to appropriate player
                            if track_id == far_player_id:
                                far_kps = abs_kps
                            elif track_id == near_player_id:
                                near_kps = abs_kps

        # 4. Pad missing poses with -1 placeholders
        if far_kps is None:
            far_kps = np.full((17, 3), -1.0)  # 17 keypoints, 3 values each (y, x, score)
        if near_kps is None:
            near_kps = np.full((17, 3), -1.0)
            
        # 5. Store frame data with both players
        all_frame_data.append({
            "frame_id": frame_count,
            "far_player_kps": far_kps,
            "near_player_kps": near_kps
        })

        frame_count += 1

    video.release()
    print(f"✅ Dual-player pose extraction complete. Processed {frame_count} frames.")
    return all_frame_data, frame_count


def create_feature_vectors(raw_data_list, frame_count):
    """Create unified feature vectors from dual-player pose data."""
    print("Engineering unified features from dual-player pose data...")
    
    # Create a dictionary for fast frame-to-data lookup
    data_lookup = {item['frame_id']: item for item in raw_data_list}
    
    all_feature_data = []
    
    # Loop through all frames processed in the video
    for frame_id in range(frame_count):
        if frame_id in data_lookup and (frame_id - 1) in data_lookup:
            current_data = data_lookup[frame_id]
            prev_data = data_lookup[frame_id - 1]
            
            # Extract keypoints for both players
            current_far_kps = current_data['far_player_kps']
            current_near_kps = current_data['near_player_kps']
            prev_far_kps = prev_data['far_player_kps']
            prev_near_kps = prev_data['near_player_kps']
            
            # Create feature vectors for each player
            far_features = create_player_features(current_far_kps, prev_far_kps)
            near_features = create_player_features(current_near_kps, prev_near_kps)
            
            # Concatenate features from both players into unified vector
            unified_feature_vector = np.concatenate([far_features, near_features])
            
            all_feature_data.append({
                "frame_id": frame_id,
                "features": unified_feature_vector
            })

    print(f"✅ Unified feature engineering complete. Processed {len(all_feature_data)} frames with features.")
    return all_feature_data


def create_player_features(current_kps, prev_kps):
    """Create feature vector for a single player (position + velocity)."""
    # Check if player data is missing (padded with -1s)
    if current_kps[0, 0] == -1 or prev_kps[0, 0] == -1:
        # Return feature vector of -1s with correct shape
        # Each keypoint contributes 4 features: pos_x, pos_y, vel_x, vel_y
        # 17 keypoints * 4 features = 68 features per player
        return np.full(68, -1.0)
    
    # Calculate velocity (change in position)
    velocity = current_kps[:, :2] - prev_kps[:, :2]
    
    # Create feature vector: [pos_x, pos_y, vel_x, vel_y] for each keypoint
    # Flatten to create single feature vector for this player
    feature_vector = np.concatenate([current_kps[:, :2].flatten(), velocity.flatten()])
    
    return feature_vector


def get_label_for_frame(frame_id, df, fps=30):
    """
    Determines the multi-class label for a frame.
    Returns:
        0 (INACTIVE): Frame is not within any point
        1 (SERVE_MOTION): Frame is within first 1.5 seconds of a point
        2 (RALLY): Frame is within a point but after the first 1.5 seconds
    """
    serve_duration_frames = int(1.5 * fps)  # 1.5 seconds in frames
    
    for _, row in df.iterrows():
        start_frame = row['start_frame']
        end_frame = row['end_frame']
        
        if start_frame <= frame_id <= end_frame:
            # Frame is within a point
            if frame_id <= start_frame + serve_duration_frames:
                return 1  # SERVE_MOTION (first 1.5 seconds)
            else:
                return 2  # RALLY (after first 1.5 seconds)
    
    return 0  # INACTIVE (not within any point)


def create_labeled_sequences(feature_data, csv_path, fps=30):
    """Create labeled sequences for training."""
    print("Creating training sequences...")
    
    # Load timestamps
    timestamps_df = pd.read_csv(csv_path)
    
    # Convert timestamps to frame numbers
    timestamps_df['start_frame'] = timestamps_df['start_time'] * fps
    timestamps_df['end_frame'] = timestamps_df['end_time'] * fps
    
    SEQUENCE_LENGTH = 60  # 60 frames = 2 seconds at 30fps
    STEP = 15             # Create a new sequence every 0.5 seconds
    
    X = []
    y = []
    
    # Create a lookup dictionary for features
    feature_lookup = {item['frame_id']: item['features'] for item in feature_data}
    max_frame = max(feature_lookup.keys())

    for i in range(0, max_frame - SEQUENCE_LENGTH, STEP):
        sequence = []
        is_valid_sequence = True
        for j in range(i, i + SEQUENCE_LENGTH):
            if j in feature_lookup:
                sequence.append(feature_lookup[j])
            else:
                # If a frame is missing features, this sequence is invalid
                is_valid_sequence = False
                break
        
        if is_valid_sequence:
            X.append(sequence)
            middle_frame_id = i + (SEQUENCE_LENGTH // 2)
            y.append(get_label_for_frame(middle_frame_id, timestamps_df, fps))

    X = np.array(X)
    y = np.array(y)

    # Print label distribution
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: 'INACTIVE', 1: 'SERVE_MOTION', 2: 'RALLY'}
    print(f"✅ Created multi-class training data. Shape of X: {X.shape}, Shape of y: {y.shape}")
    print("📊 Label distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"  {label} ({label_names.get(label, 'UNKNOWN')}): {count} sequences ({percentage:.1f}%)")
    
    return X, y


def process_video(video_path, csv_path, output_path):
    """Main processing function that orchestrates the entire pipeline."""
    print(f"🎾 Starting tennis video processing pipeline...")
    print(f"Video: {video_path}")
    print(f"Annotations: {csv_path}")
    print(f"Output: {output_path}")
    
    # Step 1: Load models
    yolo_model, movenet_model, movenet_input_size = load_models()
    
    # Step 2: Extract raw pose data
    raw_data, frame_count = extract_raw_data(video_path, yolo_model, movenet_model, movenet_input_size)
    
    # Step 3: Create feature vectors
    feature_data = create_feature_vectors(raw_data, frame_count)
    
    # Step 4: Create labeled sequences
    X, y = create_labeled_sequences(feature_data, csv_path)
    
    # Step 5: Save processed data
    print(f"💾 Saving processed data to {output_path}...")
    np.savez_compressed(output_path, X=X, y=y)
    print("✅ Data saved successfully!")
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Preprocess tennis video to extract pose data for training.")
    parser.add_argument("--video_path", required=True, help="Path to the input video file.")
    parser.add_argument("--csv_path", required=True, help="Path to the annotations CSV file.")
    parser.add_argument("--output_path", required=True, help="Path to save the output .npz file.")
    args = parser.parse_args()

    # Call the main processing function
    process_video(args.video_path, args.csv_path, args.output_path)


if __name__ == "__main__":
    main()
