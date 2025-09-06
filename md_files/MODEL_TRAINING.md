# Tennis Point Detection - Data Loading and Model Training

This directory contains components for loading tennis data and training an LSTM model to detect points in tennis videos.

## Components

### 1. TennisDataset (`tennis_dataset.py`)
A PyTorch Dataset class that:
- Loads feature vectors from `.npy` files
- Parses annotation CSV files for ground truth labels
- Creates 150-frame sequences for LSTM input
- Handles data normalization and missing values

### 2. LSTM Models (`tennis_lstm_model.py`)
Two LSTM model implementations:
- `TennisPointLSTM`: Basic LSTM with fully connected layers
- `TennisPointLSTMWithAttention`: LSTM with attention mechanism

### 3. Training Script (`train_tennis_lstm.py`)
Script to train the LSTM model with configurable parameters.

### 4. Inference Script (`inference_tennis_lstm.py`)
Script to run inference on new video features using a trained model.

## Usage

### Training
```bash
python train_tennis_lstm.py \
    --data_dir /path/to/feature/vectors \
    --annotations_dir /path/to/annotations \
    --model_save_path tennis_point_lstm.pth \
    --batch_size 32 \
    --num_epochs 50
```

### Inference
```bash
python inference_tennis_lstm.py \
    --model_path tennis_point_lstm.pth \
    --features_file /path/to/video_features.npy \
    --output_file predictions.csv
```

## Data Format

### Feature Vectors
- Shape: `(num_frames, 288)`
- Each frame contains 288 features from the DataProcessor
- Missing data represented as -1.0

### Annotations
CSV files with columns:
- `start_frame`: Start frame of point
- `end_frame`: End frame of point
- `label`: Label for the point (e.g., "point")

## Model Architecture

The default LSTM model has:
- Input size: 288 (feature vector size)
- Hidden size: 128
- LSTM layers: 2
- Dropout: 0.2
- Output: 2 classes (point/no point)

## Sequence Processing

- Sequence length: 150 frames (15 seconds at 10fps)
- Label assigned to middle frame of each sequence
- Sliding window approach with 1-frame stride