# RallyVision Tennis Point Detector

CLI tool to detect tennis points from a raw match video and output a segmented video + CSV.

## Install

```bash
pip install .
```

## Configure

Edit `config.toml` (or point `--config` to your own):

```toml
[run]
video_path = "raw_videos/your_match.mp4"
output_dir = "output_videos"
csv_output_dir = "output_csvs"
yolo_model = "nano"      # nano | small | medium | large
yolo_device = "mps"      # cpu | cuda | mps
write_csv = true
segment_video = true
```

Ensure `models/` contains `lstm_300_v0.1.pth` and `scaler_300_v0.1.joblib` (see `models/README.md`). YOLO weights auto-download.

## Run

```bash
rallyvision --config config.toml
```

Or override inline:

```bash
rallyvision --video raw_videos/your_match.mp4 --output-dir output_videos --yolo-size nano --yolo-device mps
```

Outputs:
- `<video_stem>_segments.csv` in `csv_output_dir`
- `<video_stem>_segmented.mp4` in `output_dir`
