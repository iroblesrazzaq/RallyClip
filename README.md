# RallyVision Tennis Point Detector

CLI and GUI tool that extracts points from full tennis match video and outputs a segmented video (and optional CSV).

## Prereqs
- Python 3.10-3.11
- A clean virtual environment is recommended (`python -m venv .venv && source .venv/bin/activate` or conda equivalent).

## Install
```bash
git clone https://github.com/iroblesrazzaq/RallyVision.git
cd RallyVision
pip install .
```

Models: ensure `models/` contains `lstm_300_v0.1.pth` and `scaler_300_v0.1.joblib` (see `models/README.md`). YOLO weights auto-download into `models/`.

## GUI (local)
- Run: `rallyvision gui` (auto-picks a free localhost port, opens your browser). `Ctrl-C` to quit.
- For request logs, use `RALLYVISION_GUI_VERBOSE=1 rallyvision gui`.
- Drag/drop an MP4 (recommended ≥1080p, 720p minimum). Outputs live under `~/RallyVisionJobs/<job_id>` and are downloadable from the UI.
- Advanced settings mirror `config.toml` (write CSV, thresholds, YOLO size/device, min duration, start/duration); defaults are pre-filled and should usually be left unchanged.

## Quick run (minimal)
Only the video path is required; outputs default to `./output_videos` and CSV is off by default.
```bash
rallyvision --video "raw_videos/your_match.mp4"
```
- Segmented video: `output_videos/<video_stem>_segmented.mp4`
- CSV (if enabled): `output_csvs/<video_stem>_segments.csv` or the video’s directory

## Input video quality
- Recommended source resolution: at least 720p; 1080p works best and matches the pose model’s training data. Lower resolutions tend to degrade keypoint quality and downstream segmentation accuracy.

## Common CLI flags
- `--video PATH` (required unless in config)  
- `--output-dir PATH` (default: `./output_videos`)
- `--csv-output-dir PATH` (default: video directory; enable CSV with `--write-csv`)
- `--write-csv / --no-csv` (default: off)
- `--yolo-size {nano,small,medium,large}` (default: small)
- `--yolo-device {cpu,cuda,mps}` (force pose model device)
- Thresholds/hyperparams: `--conf`, `--low`, `--high`, `--sigma`, `--seq-len`, `--overlap`, `--min-dur-sec`, `--fps`
- Model overrides: `--model-path`, `--scaler-path`
- Config file: `--config path/to/config.toml` (defaults to `./config.toml` if present)

## Config file (config.toml)
Use a TOML file instead of long CLI flags:
```toml
[run]
video_path = "raw_videos/your_match.mp4"   # change this
output_dir = "output_videos"               # change if desired
csv_output_dir = "output_csvs"             # optional; defaults to video directory

write_csv = false                          # default is off
segment_video = true
yolo_model = "nano"                        # nano | small | medium | large
yolo_device = "mps"                        # cpu | cuda | mps

# Optional overrides (usually leave as-is):
# model_path = "models/lstm_300_v0.1.pth"
# scaler_path = "models/scaler_300_v0.1.joblib"

# Postprocessing parameters: 
# to make the model more sensitive (i.e. detect more points from same signal)
# decrease high, slightly decrease low; vice-versa.
low = 0.45
high = 0.8



# Other parameters: (usually leave as-is)
sigma = 1.5 # for temporal signal gaussian smoothing, not super relevant
fps = 15.0 # do not modify
seq_len = 300 # do not modify
overlap = 150 # probably don't modify
min_dur_sec = 0.5
conf = 0.25 # yolo pose confidence threshold, .25 is well tuned
start_time = 0
duration = 999999
```
Run with:
```bash
rallyvision --config config.toml
```
