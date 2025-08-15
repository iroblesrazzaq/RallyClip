import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
VIDEO_PATH = "raw_videos/Monica Greene unedited tennis match play.mp4"
OUTPUT_VIDEO_PATH = "sanity_check_output.mp4"
SECONDS_TO_PROCESS = 60 # Process the first 30 seconds

print("Loading models...")
# Load YOLOv8 for person detection
yolo_model = YOLO('yolov8n.pt')

# Load MoveNet from TensorFlow Hub and get the inference function
movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet_infer = movenet_model.signatures['serving_default']
MOVENET_INPUT_SIZE = 192

# --- Drawing Configuration ---
# Dictionary to define connections between keypoints to draw the skeleton
SKELETON_EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    """Draws keypoints on the frame."""
    y, x, _ = frame.shape
    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx * x), int(ky * y)), 4, (0, 255, 0), -1)

def draw_skeleton(frame, keypoints, confidence_threshold=0.3):
    """Draws the skeleton on the frame."""
    y, x, _ = frame.shape
    for edge, color in SKELETON_EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        
        if c1 > confidence_threshold and c2 > confidence_threshold:
            cv2.line(frame, (int(x1 * x), int(y1 * y)), (int(x2 * x), int(y2 * y)), (0, 255, 255), 2)

def draw_keypoints_absolute(frame, keypoints, color, confidence_threshold=0.3):
    """Draws keypoints on the frame using absolute coordinates."""
    height, width, _ = frame.shape
    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx * width), int(ky * height)), 4, color, -1)

def draw_skeleton_absolute(frame, keypoints, color, confidence_threshold=0.3):
    """Draws the skeleton on the frame using absolute coordinates."""
    height, width, _ = frame.shape
    for edge, _ in SKELETON_EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        
        if c1 > confidence_threshold and c2 > confidence_threshold:
            cv2.line(frame, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), color, 2)

# --- Main Processing Loop ---
video = cv2.VideoCapture(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_limit = int(fps * SECONDS_TO_PROCESS)

# Setup video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

print(f"🏃‍♂️ Starting dual-player video processing for {SECONDS_TO_PROCESS} seconds...")
frame_count = 0

# Initialize player role assignments for tracking
far_player_id = None
near_player_id = None

while video.isOpened() and frame_count < frame_limit:
    success, frame = video.read()
    if not success:
        break

    # 1. Track players with YOLO (using tracking instead of detection)
    results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)

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
            print(f"🎾 Assigned player roles: Far player ID={far_player_id} (Blue), Near player ID={near_player_id} (Green)")
        elif len(players) == 1:
            # Only one player detected, assign as far player for now
            far_player_id = players[0][0]
            print(f"🎾 Only one player detected: Far player ID={far_player_id} (Blue)")

    # 3. Process and visualize both players
    if results[0].boxes is not None and results[0].boxes.id is not None:
        for i, box in enumerate(results[0].boxes):
            if box.id is not None:
                track_id = int(box.id.cpu().numpy()[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Determine player color and label
                if track_id == far_player_id:
                    color = (255, 0, 0)  # Blue for far player
                    label = f"Far Player (ID: {track_id})"
                elif track_id == near_player_id:
                    color = (0, 255, 0)  # Green for near player
                    label = f"Near Player (ID: {track_id})"
                else:
                    color = (128, 128, 128)  # Gray for unassigned players
                    label = f"Unassigned (ID: {track_id})"
                
                # Draw bounding box with player-specific color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label above bounding box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Extract and draw pose if this is one of our assigned players
                if track_id == far_player_id or track_id == near_player_id:
                    player_crop = frame[y1:y2, x1:x2]
                    if player_crop.shape[0] > 0 and player_crop.shape[1] > 0:
                        # Prepare image for MoveNet
                        image_for_movenet = tf.image.resize_with_pad(
                            tf.expand_dims(player_crop, axis=0), MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE
                        )
                        outputs = movenet_infer(tf.cast(image_for_movenet, dtype=tf.int32))
                        keypoints_relative = outputs['output_0'][0, 0].numpy()

                        # Convert relative keypoints to absolute coordinates for drawing on full frame
                        keypoints_absolute = np.zeros_like(keypoints_relative)
                        keypoints_absolute[:, 0] = keypoints_relative[:, 0] * (y2 - y1) / height + y1 / height  # y-coordinate
                        keypoints_absolute[:, 1] = keypoints_relative[:, 1] * (x2 - x1) / width + x1 / width   # x-coordinate
                        keypoints_absolute[:, 2] = keypoints_relative[:, 2]  # confidence score

                        # Draw skeleton and keypoints on the full frame
                        draw_keypoints_absolute(frame, keypoints_absolute, color)
                        draw_skeleton_absolute(frame, keypoints_absolute, color)

    # Write the annotated frame to the output video
    out.write(frame)
    frame_count += 1
    
    # Optional: Display progress
    if frame_count % int(fps) == 0:
        print(f"  Processed {frame_count // fps} seconds...")

# Release everything
video.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Sanity check complete. Annotated video saved to: {OUTPUT_VIDEO_PATH}")
