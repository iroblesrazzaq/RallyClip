# RallyClip

RallyClip is an open-source CLI for tennis video segmentation. It extracts rally/point intervals from full match footage and outputs a segmented video plus optional CSV timestamps.

This public repo is the local product:
- CLI inference/runtime
- local GUI code
- open training pipeline code

The hosted production app lives in the private `rallyclip-prod` repo. Training data and private evaluation assets are not part of this repository.

## Prereqs
- Python 3.10+
- A clean virtual environment is recommended:
  `python -m venv .venv && source .venv/bin/activate` (or conda equivalent)

## Install
```bash
git clone https://github.com/iroblesrazzaq/RallyClip.git
cd RallyClip
pip install .
```

## Model assets
RallyClip model artifacts live under `models/rallyclip_v0.3.1/`:
- `model.onnx`
- `scaler.json`
- `manifest.json`

YOLO pose weights are downloaded automatically into `models/` when needed.

## Quick run (minimal CLI)
Only the video path is required; segmented output defaults to `./output_videos`.

```bash
rallyclip --video "raw_videos/your_match.mp4"
```

- Segmented video: `output_videos/<video_stem>_segmented.mp4`
- CSV (if enabled): `output_csvs/<video_stem>_segments.csv` or the input video directory

## Input video quality
- Recommended source resolution: at least 720p
- 1080p works best and matches pose-model training assumptions
- Lower resolutions can reduce keypoint quality and segmentation accuracy

## Common CLI flags
- `--video PATH` (required unless supplied in config)
- `--output-dir PATH` (default: `./output_videos`)
- `--csv-output-dir PATH` (default: video directory; enable CSV with `--write-csv`)
- `--write-csv / --no-csv` (default: off)
- `--yolo-size {nano,small,medium,large}` (default: `small`)
- `--yolo-device {cpu,cuda,mps}` (force pose model device)
- Advanced overrides: `--conf`, `--low`, `--high`, `--sigma`, `--seq-len`, `--overlap`, `--min-dur-sec`, `--fps`
- Artifact overrides: `--artifact-dir`, `--manifest-path`
- Config file: `--config path/to/config.toml` (defaults to `./config.toml` if present)

## Config file (`config.toml`)
Use TOML config instead of long CLI invocations:

```toml
[run]
video_path = "raw_videos/your_match.mp4"   # required
output_dir = "output_videos"
csv_output_dir = "output_csvs"             # optional; defaults to video directory

write_csv = false
segment_video = true
yolo_model = "nano"                        # nano | small | medium | large
yolo_device = "mps"                        # cpu | cuda | mps

# Optional artifact overrides:
# artifact_dir = "models/rallyclip_v0.3.1"

# Postprocessing / inference parameters
low = 0.45
high = 0.7
sigma = 1.0
min_dur_sec = 1.0

# Temporal settings for v0.3.1 defaults
fps = 5.0
seq_len = 100
overlap = 50
conf = 0.25
start_time = 0
duration = 999999
```

Run with:
```bash
rallyclip --config config.toml
```

## Training pipeline (dev)
Training uses a separate YAML config and a step-based pipeline.

- Config: `configs/train/base.yaml`
- Entry point: `python train.py --config configs/train/base.yaml`
- Docs: `docs/training.md`

The training code is public. Datasets, private evaluation sets, and any training secrets or paid infrastructure remain private.
