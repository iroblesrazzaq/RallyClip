# RallyClip

RallyClip is an open-source tool for tennis video segmentation. It extracts rally/point intervals from full match footage and outputs a segmented video plus optional CSV timestamps.

This repo ships:
- `rallyclip` CLI for local inference
- `rallyclip-desktop` native desktop app (pywebview: WKWebView on macOS, WebView2 on Windows)
- `rallyclip gui` browser-based local UI for development
- open training pipeline code

## Current release status

The first public desktop release is `v0.1.0` for **Apple Silicon macOS only**.
It was built, signed, notarized, stapled, and uploaded manually as a DMG. There
is not yet a full GitHub Actions release pipeline.

The packaged app uses the native QtMultimedia replay viewer for saved matches.
Browser/WebM playback remains a development fallback.

## Runtime architecture direction

The `refactor/runtime-api-engine` branch is splitting the runtime into:

- `rallyclip_core`: pure contracts, interval helpers, pipeline selection, saved
  playback manifests, and source-time scheduler rules.
- `rallyclip_engine`: analysis execution. A model pipeline owns preprocessing,
  inference, postprocessing, and CSV/video-ready output.
- `rallyclip_api`: application service layer that Flask, CLI, desktop, and future
  clients can share.
- UI clients: native macOS, browser dev UI, and future mobile clients own their
  own video rendering and controls.

See `docs/ENVIRONMENT.md` and `docs/runtime-api-engine-refactor.md` for the current
branch handoff and test commands.


## Features coming soon (in rough order)
- gh release (desktop app)
- Retrain on YOLOv26-pose model for better accuracy/lower FLOPS
- Retrain on scaled data
- quantize yolov26 to int8 for edge inference speed


## Features down the road
In no particular order,
- OPTIMIZE TF OUTTA THE MODELS TO MAKE THEM FASTER AND BETTER!!!!
- Find best inference setup (YOLO batching, etc) – mostly for deployed app by also will release open source :)
- port to TS, npm for easier runtime, install with npm, no dealing with python deps
- Doubles support (need to label + train on doubles data)
- Open dataset (once deployed, opt-in for publicly available dataset for open-source community to use)
- Match scoring
- Mobile app (once I can scale data more to push down model size by expanding the repertoire of architectures I can use, particularly in training more complex deep learning models from scratch)
- Better resolution support: 720p, 1080p, 1440p, 4K. Training models for each, finding out best yolo params etc for them, given that YOLO downscales regardless it might not matter, but will look into. 

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

### Desktop app
```bash
pip install ".[desktop]"
rallyclip-desktop
```

The desktop app bundles the local Flask backend in a native window. Device selection defaults to **Auto** (`CUDA > MPS > CPU`) and can be overridden in Advanced settings.

### Browser GUI (development)
```bash
rallyclip gui
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
- `--yolo-device {cpu,cuda,mps,coreml}` (force pose model device; `coreml` runs the bundled static ONNX on the Apple Neural Engine — several times faster pose on Apple silicon, opt-in because `cpu` is the byte-parity reference)
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
yolo_device = "mps"                        # cpu | cuda | mps | coreml

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

## GitHub Releases

The current `v0.1.0` macOS release was created manually. GitHub Releases hosts
the DMG exactly as uploaded; GitHub does not convert zips or app bundles into a
DMG.

Future work: add GitHub Actions CI/CD for tests, app build, Apple signing,
notarization, stapling, DMG creation, and release upload.

To cut a release locally:
```bash
pip install ".[desktop,pack]"
pyinstaller --noconfirm RallyClip.spec
```

The runtime is torch-free: pose inference runs on the ONNX bundled in
`models/rallyclip_v0.3.1/` via onnxruntime (`extraction/yolo_onnx_runner.py`).
Training and the legacy .pt path need `pip install ".[train]"`.

### Headless mode
The shipped binary can run the full pipeline without launching the GUI. Pass
`--cli` as the first argument; everything after it is the regular `rallyclip`
CLI:

```bash
dist/RallyClip/RallyClip --cli --video match.mp4 --start-time 1240 --duration 180 \
  --write-csv --csv-output-dir /tmp/out --no-segment-video
```

`RallyClip --cli --help` prints the full flag reference. Note: YOLO pose
weights are not bundled; the first headless run downloads them into a
`models/` folder under the current working directory.
