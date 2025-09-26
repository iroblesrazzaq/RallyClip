# DeepMatch Tennis Point Detector

This tool automates the process of finding and extracting points from a raw tennis match video.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Add Model Assets:**
    Place your trained model files (`best_model.pth`, `scaler.joblib`, and `yolov8n-pose.pt`) into the `models/` directory. See `models/README.md` for details.
    
## Usage

Run the main script with your video. The model and scaler default to the seq_len=300 assets.

```bash
python detect_points.py \
  --video /path/to/your/match.mp4 \
  --output-dir output_videos
```

Optional overrides:

```bash
# Defaults:
# --model defaults to checkpoints/seq_len300/best_model.pth
# --scaler defaults to data/seq_len_300/scaler.joblib
# --yolo-model defaults to yolov8s-pose.pt (resolved under models/)
# --output-dir defaults to output_videos

python detect_points.py \
  --video /path/to/your/match.mp4 \
  --model /custom/path/best_model.pth \
  --scaler /custom/path/scaler.joblib \
  --yolo-model yolov8s-pose.pt \
  --output-dir output_videos
```

## Outputs

The script will generate two files in the specified output directory:

- `match_segments.csv`: A CSV file with the start and end times of each detected point.
- `match_segmented.mp4`: A new video file containing only the clips of the detected points.

