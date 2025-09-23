Tennis Tracker: Free, open source project to automate cutting dead time between points for easier point review. 

Implementation: Using YOLOv8, track player bounding boxes to estimate their poses with movenet, 
then use movenet pose features, as well as engineered features (velocity, acceleration, etc) for both players
to feed into an LSTM that returns the confidence that a given frame (given the past n frames) is during a point or not. 

## Data Processing Pipeline

The pipeline now includes a `DataProcessor` class that handles:
1. Player assignment using heuristic logic (near/far player detection)
2. Feature engineering including centroid, velocity, and acceleration calculation for both player and keypoints
3. Biomechanical features including limb lengths for anatomically connected joints
4. Creation of LSTM-ready feature vectors with proper handling of missing data

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
  - 34 elements: Keypoint velocity components [vx1, vy1, ..., vx17, vy17]
  - 34 elements: Keypoint acceleration components [ax1, ay1, ..., ax17, ay17]
  - 14 elements: Limb lengths (anatomically connected joints)

Missing players are represented with -1 values for position data and 0 for velocity/acceleration, which are outside the valid coordinate range [0, width/height].

## Optimized Pose Extractor

An optimized version of the pose extractor is available that leverages Apple MPS parallel inference capabilities. 
See [POSE_EXTRACTOR_OPTIMIZED.md](POSE_EXTRACTOR_OPTIMIZED.md) for details.

Current:
- iterate on model to improve accuracy: hyperparameters, data aug, annotate more data?
- refine postprocessing steps to get actual point start/ends, quantify performance for that

Later:
look into LSD instead of Hough for line detection, further court detection optimizations: not super necessary, court detection is good enough for now, not bottleneck

inference optimizatoin: 
- look into finetuning yolov8n pose on tennis specific dataset from yolov8 pose large to optimize for inference speed
- look into batch yolo processing
- chunking inference frame data write to saved npz for memory

Look into data augmentation with visual language models for labelling data. gemma-3-27b-it, qwen 2.5 VL 32b?

Done:
August 27, 2025:
- working on court detection for bounding box masking to only capture the players in the relevant playing area
- issue - players blocking lines in randomly selected frame -> incomplete baseline (mostly baseline suffers from this issues) - what to do?
    - can we sample a few frames and overlay all their candidate lines for this?
Sept 11: MVP complete


The point is to keep it local - web frontend for easier js gui development, but to make it free, we use the local machine. 

My next steps are: 1. fully productionalizing - dockerize, test, etc, then publish. 
2. improve model performance- increase annotated data, use data augmentation techniques, investigate other model architectures (lstm with attention), look into more sophisticating post processing steps (HMM or bayesian approach or even DL approach). 
3. optimize performance - stepwise saving for pose extraction to reduce memory load of all npz frames, look into batched YOLO inference, later on maybe finetune yolov8nano model on large to optimize inference speed. maybe even later use tiling to get second player more consistently