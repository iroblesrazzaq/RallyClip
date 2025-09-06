# Handling Missing Player Data in Feature Vectors

## Problem
When a player is not detected in a frame, we need to represent this in our feature vector. Using 0 is problematic because 0 is a valid coordinate value.

## Solution
We use `-1` to represent missing player data in the feature vectors, which is outside the valid coordinate range [0, width/height].

For velocity and acceleration calculations, we use `0` to represent no measurable movement, since:
- When a player is not detected, there is no measurable movement during that frame
- Velocity and acceleration can be negative for present players, so 0 is a valid value for actual movement

## Implementation Details

### Feature Vector Structure
- Total size: 288 elements (144 per player × 2 players)
- Per player structure:
  - 1 element: Presence indicator (1.0 = present, -1.0 = absent)
  - 4 elements: Bounding box coordinates [x1, y1, x2, y2]
  - 2 elements: Centroid coordinates (center_x, center_y)
  - 2 elements: Player velocity components (vx, vy)
  - 2 elements: Player acceleration components (ax, ay)
  - 34 elements: Keypoint coordinates [x1, y1, x2, y2, ..., x17, y17]
  - 17 elements: Keypoint confidence scores
  - 34 elements: Keypoint velocity components [vx1, vy1, vx2, vy2, ..., vx17, vy17]
  - 34 elements: Keypoint acceleration components [ax1, ay1, ax2, ay2, ..., ax17, ay17]
  - 14 elements: Limb lengths (anatomically connected joints)

### Examples

#### Both players present with consecutive detections:
```
[1.0, x1, y1, x2, y2, center_x, center_y, player_vx, player_vy, player_ax, player_ay,
 kp_x1, kp_y1, ..., kp_x17, kp_y17, kp_conf1, ..., kp_conf17,
 kp_vx1, kp_vy1, ..., kp_vx17, kp_vy17, kp_ax1, kp_ay1, ..., kp_ax17, kp_ay17,
 limb_len1, limb_len2, ..., limb_len14,
 1.0, x1, y1, x2, y2, center_x, center_y, player_vx, player_vy, player_ax, player_ay,
 kp_x1, kp_y1, ..., kp_x17, kp_y17, kp_conf1, ..., kp_conf17,
 kp_vx1, kp_vy1, ..., kp_vx17, kp_vy17, kp_ax1, kp_ay1, ..., kp_ax17, kp_ay17,
 limb_len1, limb_len2, ..., limb_len14]
```

#### No players present:
```
[-1, -1, -1, ..., 0, 0, 0, 0, -1, -1, ..., -1, 0, 0, ..., 0, 0, ..., 0, -1, -1, ..., -1] (288 elements)
Note: Position data set to -1, velocity/acceleration set to 0, limb lengths set to -1
```

#### Only near player present:
```
[1.0, x1, y1, x2, y2, center_x, center_y, player_vx, player_vy, player_ax, player_ay,
 kp_x1, kp_y1, ..., kp_x17, kp_y17, kp_conf1, ..., kp_conf17,
 kp_vx1, kp_vy1, ..., kp_vx17, kp_vy17, kp_ax1, kp_ay1, ..., kp_ax17, kp_ay17,
 limb_len1, limb_len2, ..., limb_len14,
 -1, -1, -1, ..., 0, 0, 0, 0, -1, -1, ..., -1, 0, 0, ..., 0, 0, ..., 0, -1, -1, ..., -1]
```

## Key Design Decisions

### Handling Missing Players
- **Position data**: Set to -1 (non-plausible coordinate)
- **Player Velocity/Acceleration**: Set to 0 (no measurable movement)
- **Keypoint data**: Set to -1 (missing data)
- **Keypoint Velocity/Acceleration**: Set to 0 (no measurable movement)
- **Limb lengths**: Set to -1 (missing data)

### Handling Intermittent Detections
Players frequently leave and re-enter the detection area. Our approach:
- **Velocity/Acceleration only calculated for consecutive detections**
- **No interpolation or filling of missing positions**
- **When players reappear, velocity is calculated from their new position**

### Example Scenario
```
Frame 1: Player detected at (100, 200) -> Store position
Frame 2: Player not detected -> Player velocity = 0, Keypoint velocity = 0, Limb lengths = -1
Frame 3: Player detected at (150, 220) -> Player velocity = (50, 20), Keypoint velocity = calculated, Limb lengths = calculated
```

## Benefits of This Approach

1. **Realistic modeling**: Doesn't assume player positions during undetected frames
2. **Semantic clarity**: Different values for different meanings (-1 = missing, 0 = no movement)
3. **No false interpolation**: Doesn't create artificial movement data
4. **Robust to player exits**: Handles players leaving/re-entering court naturally
5. **Detailed body movement**: Keypoint velocity/acceleration captures detailed body dynamics
6. **Biomechanical features**: Limb lengths capture anatomical relationships
7. **PyTorch compatibility**: Can be fed directly into LSTM tensors

## Keypoint Velocity/Acceleration Benefits

1. **Body dynamics**: Captures movement of individual body parts
2. **Action recognition**: Helps identify specific tennis actions (serve, forehand, backhand)
3. **Injury prevention**: Can detect abnormal movement patterns
4. **Technique analysis**: Provides detailed feedback on player technique

## Limb Length Benefits

1. **Pose stability**: Captures consistent body proportions
2. **Action recognition**: Different tennis strokes have characteristic limb configurations
3. **Biomechanical analysis**: Provides insight into player technique and form
4. **Noise reduction**: More stable than raw keypoint coordinates
5. **Anatomical meaning**: Each feature represents a real physical connection in the body

## Preprocessing for ML Models

The feature vectors can be fed directly to PyTorch LSTM models. However, you may want to consider:

1. **Normalization**: Normalize coordinates, velocity, and acceleration appropriately
2. **Model learning**: The model can learn to interpret -1 as "missing data" and 0 as "no movement"

Example usage with PyTorch:
```python
import torch
import torch.nn as nn

# Direct usage - no preprocessing needed
feature_tensor = torch.FloatTensor(feature_vector)

# If using in a sequence
sequence = torch.FloatTensor([feature_vector1, feature_vector2, ...])
lstm = nn.LSTM(input_size=288, hidden_size=64, num_layers=2)
output, (hidden, cell) = lstm(sequence)
```