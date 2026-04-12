# RallyClip

RallyClip is an open-source CLI for tennis video segmentation. It extracts rally/point intervals from full match footage and outputs a segmented video plus optional CSV timestamps.

This public repo is the local product:
- CLI inference/runtime
- local GUI code
- open training pipeline code

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
## Features coming soon (in rough order)
- Deployed model to web (alpha coming soon!)
- Scaling data -> better models
- Rally segmentation

## Features down the road
In no particular order,
- port to TS, npm for easier runtime, install with npm, no dealing with python deps
- Doubles support (need to label + train on doubles data)
- Open dataset (once deployed, opt-in for publicly available dataset for open-source community to use)
- Match scoring
- Mobile app (once I can scale data more to push down model size by expanding the repertoire of architectures I can use, particularly in training more complex deep learning models from scratch)

## Optimizations/things I'm thinking about
These aren't necessarily features per se, but things I want to try implementing for better + more efficient models
- Training postprocessing model (thinking a 1d-CNN for now on the LSTM outputs)
- Adding some lower-dimensional visual representation of the court to the postprocessing model
  - maybe some trained/fine-tuned vision conv net to extract features missed in the pose model
- Training frame-probability objective and post-process objective simultaneously
  - The way I envision this is either:
    1. Just one loss term of a combo of IoU over frames + weighted MSE distance of endpoints biased towards longer points rather than underestimating lengths, as well as losses for false positives or false negatives
    2. Two loss terms, one being the frame-level probability objective used before, and the other is the loss term described above, the frame level loss would only impact LSTM backbone, the segment loss would backprop over entire net
- Low confidence moments for manual review (idea right now is to train the model on an additional loss term for confidence, likely both in LSTM frame backbone and output postprocess
- Fine tune YOLO? One option is to fine-tune YOLO nano on YOLO Large outputs for all match footage. Or some other less naive approach with the models in mind, potentially including YOLO outputs in backprop (attaching YOLO directly to LSTM model as a CNN input transformation, then doing LoRA or last 2 layers adaptation only)
- exploring different model architectures - maybe LSTM with attention
- Fine-tune YOLO models to only capture 1 of: near player, far player, or the ball itself.
- Adding audio modality (is clear, helpful signal sometimes, very unhelpful other times)
  - would need different models if audio doesn't exist, as well as audio preprocessing pipeline. Need to train to be robust to noise.
- More data augmentaion - currently only doing mirror video
- Feature engineering - experiment more. 
