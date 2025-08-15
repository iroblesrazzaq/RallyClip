#!/usr/bin/env python3
"""
Tennis Action Recognition Prediction Script

This script uses a trained LSTM model to analyze tennis videos and automatically
detect point boundaries for highlight reel generation. It processes videos to
identify:
- Point starts (SERVE_MOTION transitions)
- Point ends (sustained INACTIVE periods)
- Complete point timestamps for highlight extraction
"""

import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import json
from datetime import datetime

# Import preprocessing functions
from preprocessing import load_models, extract_raw_data, create_feature_vectors, create_player_features


def load_trained_model(model_path):
    """
    Load a trained Keras model for tennis action recognition.
    
    Args:
        model_path (str): Path to the trained model file
        
    Returns:
        tf.keras.Model: Loaded model ready for prediction
    """
    print(f"🧠 Loading trained model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Total parameters: {model.count_params():,}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def process_video_for_prediction(video_path):
    """
    Process a video file to extract features compatible with the trained model.
    
    Args:
        video_path (str): Path to the input video file
        
    Returns:
        tuple: (feature_data, frame_count, fps) where feature_data contains
               unified feature vectors for each frame
    """
    print(f"🎬 Processing video for prediction: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Load models (same as training pipeline)
    yolo_model, movenet_model, movenet_input_size = load_models()
    
    # Extract raw pose data for both players
    raw_data, frame_count = extract_raw_data(video_path, yolo_model, movenet_model, movenet_input_size)
    
    # Create unified feature vectors
    feature_data = create_feature_vectors(raw_data, frame_count)
    
    # Get video FPS for timestamp conversion
    import cv2
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    
    print(f"✅ Video processing complete!")
    print(f"   - Total frames: {frame_count}")
    print(f"   - Feature frames: {len(feature_data)}")
    print(f"   - Video FPS: {fps}")
    
    return feature_data, frame_count, fps


def create_prediction_sequences(feature_data, sequence_length=60, step=1):
    """
    Create overlapping sequences from feature data for dense prediction.
    
    Args:
        feature_data (list): List of feature dictionaries with frame_id and features
        sequence_length (int): Length of each sequence (default: 60 frames)
        step (int): Step size between sequences (default: 1 for dense prediction)
        
    Returns:
        tuple: (sequences, frame_ids) where sequences is array of shape 
               (n_sequences, sequence_length, n_features) and frame_ids 
               contains the center frame of each sequence
    """
    print(f"🔄 Creating prediction sequences...")
    print(f"   - Sequence length: {sequence_length} frames")
    print(f"   - Step size: {step} frame(s)")
    
    # Create feature lookup dictionary
    feature_lookup = {item['frame_id']: item['features'] for item in feature_data}
    max_frame = max(feature_lookup.keys())
    
    sequences = []
    frame_ids = []
    
    # Create overlapping sequences for dense prediction
    for i in range(0, max_frame - sequence_length + 1, step):
        sequence = []
        is_valid_sequence = True
        
        # Build sequence
        for j in range(i, i + sequence_length):
            if j in feature_lookup:
                sequence.append(feature_lookup[j])
            else:
                # Skip sequences with missing frames
                is_valid_sequence = False
                break
        
        if is_valid_sequence:
            sequences.append(sequence)
            # Store center frame ID for this sequence
            center_frame = i + (sequence_length // 2)
            frame_ids.append(center_frame)
    
    sequences = np.array(sequences)
    frame_ids = np.array(frame_ids)
    
    print(f"✅ Created {len(sequences)} prediction sequences")
    return sequences, frame_ids


def predict_video_timeline(model, sequences, frame_ids):
    """
    Use the trained model to predict action classes for the entire video timeline.
    
    Args:
        model (tf.keras.Model): Trained model
        sequences (np.ndarray): Input sequences for prediction
        frame_ids (np.ndarray): Frame IDs corresponding to each sequence
        
    Returns:
        tuple: (predictions, probabilities, timeline) where timeline maps
               frame_id to predicted class
    """
    print(f"🔮 Generating predictions for video timeline...")
    
    # Get model predictions
    probabilities = model.predict(sequences, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    
    # Create timeline mapping frame_id to prediction
    timeline = {}
    for frame_id, pred, prob in zip(frame_ids, predictions, probabilities):
        timeline[frame_id] = {
            'prediction': int(pred),
            'probabilities': prob.tolist(),
            'confidence': float(np.max(prob))
        }
    
    # Analyze prediction distribution
    unique, counts = np.unique(predictions, return_counts=True)
    label_names = {0: 'INACTIVE', 1: 'SERVE_MOTION', 2: 'RALLY'}
    
    print("📊 Prediction distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        print(f"   {label} ({label_names.get(label, 'UNKNOWN')}): {count} sequences ({percentage:.1f}%)")
    
    return predictions, probabilities, timeline


def detect_points_from_timeline(timeline, fps, min_inactive_duration=2.0, min_point_duration=1.0):
    """
    Detect tennis points using a state machine approach on the prediction timeline.
    
    Args:
        timeline (dict): Frame-to-prediction mapping
        fps (float): Video frames per second
        min_inactive_duration (float): Minimum inactive duration to end a point (seconds)
        min_point_duration (float): Minimum point duration to be valid (seconds)
        
    Returns:
        list: List of detected points with start_frame, end_frame, and timestamps
    """
    print(f"🎾 Detecting tennis points from timeline...")
    print(f"   - Minimum inactive duration: {min_inactive_duration}s")
    print(f"   - Minimum point duration: {min_point_duration}s")
    
    # Convert durations to frames
    min_inactive_frames = int(min_inactive_duration * fps)
    min_point_frames = int(min_point_duration * fps)
    
    # Sort frames for sequential processing
    sorted_frames = sorted(timeline.keys())
    
    points = []
    current_point_start = None
    inactive_streak = 0
    
    # State machine variables
    in_point = False
    
    for frame_id in sorted_frames:
        prediction = timeline[frame_id]['prediction']
        confidence = timeline[frame_id]['confidence']
        
        if prediction == 1:  # SERVE_MOTION - potential point start
            if not in_point:
                # Start new point
                current_point_start = frame_id
                in_point = True
                inactive_streak = 0
                print(f"   🟢 Point started at frame {frame_id} ({frame_id/fps:.1f}s)")
            
        elif prediction == 0:  # INACTIVE
            if in_point:
                inactive_streak += 1
                
                # Check if we've had enough consecutive inactive frames to end the point
                if inactive_streak >= min_inactive_frames:
                    # End current point
                    point_end = frame_id - inactive_streak  # End at start of inactive streak
                    point_duration_frames = point_end - current_point_start
                    
                    # Only keep points that meet minimum duration
                    if point_duration_frames >= min_point_frames:
                        point = {
                            'start_frame': current_point_start,
                            'end_frame': point_end,
                            'start_time': current_point_start / fps,
                            'end_time': point_end / fps,
                            'duration': point_duration_frames / fps
                        }
                        points.append(point)
                        print(f"   🔴 Point ended at frame {point_end} ({point_end/fps:.1f}s) - Duration: {point['duration']:.1f}s")
                    else:
                        print(f"   ⚠️ Point too short ({point_duration_frames/fps:.1f}s) - Discarded")
                    
                    # Reset state
                    in_point = False
                    current_point_start = None
                    inactive_streak = 0
            else:
                # Continue inactive streak when not in point
                inactive_streak += 1
                
        else:  # RALLY (prediction == 2)
            if in_point:
                # Reset inactive streak - we're still in the point
                inactive_streak = 0
    
    # Handle case where video ends while in a point
    if in_point and current_point_start is not None:
        final_frame = sorted_frames[-1]
        point_duration_frames = final_frame - current_point_start
        
        if point_duration_frames >= min_point_frames:
            point = {
                'start_frame': current_point_start,
                'end_frame': final_frame,
                'start_time': current_point_start / fps,
                'end_time': final_frame / fps,
                'duration': point_duration_frames / fps
            }
            points.append(point)
            print(f"   🔴 Final point ended at frame {final_frame} ({final_frame/fps:.1f}s) - Duration: {point['duration']:.1f}s")
    
    print(f"✅ Detected {len(points)} tennis points")
    return points


def format_timestamp(seconds):
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def save_results(points, output_path, video_path, model_path):
    """
    Save detection results to JSON file.
    
    Args:
        points (list): Detected points
        output_path (str): Output file path
        video_path (str): Original video path
        model_path (str): Model path used
    """
    results = {
        'metadata': {
            'video_path': video_path,
            'model_path': model_path,
            'detection_time': datetime.now().isoformat(),
            'total_points': len(points)
        },
        'points': points
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to {output_path}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Use trained model to detect tennis points and generate highlight timestamps."
    )
    parser.add_argument(
        "--video_path", 
        required=True, 
        help="Path to the input video file"
    )
    parser.add_argument(
        "--model_path", 
        required=True, 
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--output_path", 
        help="Path to save results JSON file (optional)"
    )
    parser.add_argument(
        "--min_inactive_duration", 
        type=float, 
        default=2.0, 
        help="Minimum inactive duration to end a point in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--min_point_duration", 
        type=float, 
        default=1.0, 
        help="Minimum point duration to be valid in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    print("🎾 Tennis Point Detection System")
    print("=" * 50)
    print(f"Video: {args.video_path}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_path or 'Console only'}")
    print("=" * 50)
    
    try:
        # Load trained model
        model = load_trained_model(args.model_path)
        
        # Process video to extract features
        feature_data, frame_count, fps = process_video_for_prediction(args.video_path)
        
        # Create prediction sequences
        sequences, frame_ids = create_prediction_sequences(feature_data)
        
        # Generate predictions
        predictions, probabilities, timeline = predict_video_timeline(model, sequences, frame_ids)
        
        # Detect points using state machine
        points = detect_points_from_timeline(
            timeline, fps, 
            args.min_inactive_duration, 
            args.min_point_duration
        )
        
        # Display results
        print("\n🏆 DETECTED TENNIS POINTS")
        print("=" * 50)
        
        if points:
            total_highlight_duration = sum(point['duration'] for point in points)
            
            for i, point in enumerate(points, 1):
                start_time = format_timestamp(point['start_time'])
                end_time = format_timestamp(point['end_time'])
                duration = format_timestamp(point['duration'])
                
                print(f"Point {i:2d}: {start_time} - {end_time} (Duration: {duration})")
            
            print("=" * 50)
            print(f"Total Points: {len(points)}")
            print(f"Total Highlight Duration: {format_timestamp(total_highlight_duration)}")
            print(f"Video Length: {format_timestamp(frame_count / fps)}")
            print(f"Highlight Percentage: {(total_highlight_duration / (frame_count / fps)) * 100:.1f}%")
        else:
            print("No tennis points detected in this video.")
        
        # Save results if output path specified
        if args.output_path:
            save_results(points, args.output_path, args.video_path, args.model_path)
        
        print("\n✅ Point detection completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()