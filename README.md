# RallyClip

RallyClip is an open-source tool for tennis video segmentation. It extracts rally/point intervals from full match footage and outputs a segmented video plus optional CSV timestamps.

This repo ships:
- `rallyclip` CLI for local inference
- `rallyclip-desktop` for macos desktop app
- `rallyclip gui` browser-based local UI for development
- open training pipeline code


## Features coming soon (in rough order)
- iOS mobile app beta coming soon


## Current release status

The current public desktop release is `v0.3.0` for **Apple Silicon macOS only**.
It was built, signed, notarized, stapled, and uploaded manually as a DMG. 

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



## Features down the road
In no particular order,
- OPTIMIZE TF OUTTA THE MODELS TO MAKE THEM FASTER AND BETTER!!!!
- Mobile app (once I can scale data more to push down model size by expanding the repertoire of architectures I can use, particularly in training more complex deep learning models from scratch)
- Doubles support (need to label + train on doubles data)
- Open dataset (once deployed, opt-in for publicly available dataset for open-source community to use)
- Match scoring
- Better resolution support: 720p, 1080p, 1440p, 4K. Training models for each, finding out best yolo params etc for them, given that YOLO downscales regardless it might not matter, but will look into. 



## Prereqs
- Python 3.10+
- A clean virtual environment is recommended:
  `python -m venv .venv && source .venv/bin/activate` (or conda equivalent)

## Install
```bash
git clone https://github.com/iroblesrazzaq/RallyClip.git
cd RallyClip
pip install ".[cpu]"
```

### Desktop app
```bash
pip install ".[desktop,cpu]"   # Mac / CPU-only; use ".[desktop,gpu]" on NVIDIA (see docs/ENVIRONMENT.md)
rallyclip-desktop
```

The desktop app bundles the local Flask backend in a native window. Device selection defaults to **Auto** (`CUDA > CoreML > MPS > CPU`) and can be overridden in Advanced settings. CUDA for the torch-free ONNX pose path needs the optional `[gpu]` extra (`onnxruntime-gpu`); never combine `[cpu]` and `[gpu]` — see `docs/ENVIRONMENT.md`.

### Browser GUI (development)
```bash
rallyclip gui
```

## Model assets
RallyClip model artifacts live under `models/rallyclip_v0.5.0/`:
- `model.onnx`
- `scaler.json`
- `manifest.json`
- `yolov8n-pose-960-dynamic.onnx`
- `yolov8n-pose-544x960-static.onnx`

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
- `--yolo-device {cpu,cuda,mps,coreml}` (force pose model device; on Apple silicon auto picks `coreml` — the bundled static ONNX on the Apple Neural Engine, several times faster pose — and degrades to `cpu`, the byte-parity reference, whenever CoreML is unusable)
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
# artifact_dir = "models/rallyclip_v0.5.0"

# Postprocessing / inference parameters
low = 0.45
high = 0.7
sigma = 1.0
min_dur_sec = 1.0

# Temporal settings for v0.5.0 defaults
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

Tagging `v*` on `main` runs `.github/workflows/release.yml`: tests, PyInstaller
`.app` via `RallyClip.spec`, Developer ID signing, DMG wrap, Apple notarization,
stapling, and a draft GitHub Release with `RallyClip-<version>-macOS-arm64.dmg`.
`workflow_dispatch` builds the same artifact without publishing (unsigned if
signing secrets are missing).

### One-time GitHub secrets

Repo Settings → Secrets and variables → Actions:

| Secret | What |
|---|---|
| `MACOS_CERTIFICATE_P12_BASE64` | Developer ID Application `.p12`, base64 (`base64 -i cert.p12 \| pbcopy`) |
| `MACOS_CERTIFICATE_PASSWORD` | Password for that `.p12` |
| `APPSTORE_ISSUER_ID` | App Store Connect API issuer ID |
| `APPSTORE_API_KEY_ID` | Key ID (`AuthKey_<id>.p8`) |
| `APPSTORE_API_PRIVATE_KEY` | Full contents of the `.p8` (including BEGIN/END lines) |

Optional: `MACOS_SIGN_IDENTITY` if the cert name is not
`Developer ID Application: Ismael Robles-Razzaq (L9W8X6N9B9)`.

Export the cert from Keychain Access → My Certificates → Developer ID
Application. Create the API key under App Store Connect → Users and Access →
Integrations → App Store Connect API (Developer or App Manager).

### Cut a release

1. Set `version` in `pyproject.toml` (the tag must be `v` plus that value).
2. Merge to `main`, then `git tag v0.x.y && git push origin v0.x.y`.
3. Wait for the Release workflow. Publish the draft after a Gatekeeper smoke test.

Local equivalent (macOS, cert in your login keychain):

```bash
pip install ".[desktop,pack,cpu]"
pyinstaller --noconfirm RallyClip.spec
bash scripts/release/package_macos.sh dist/RallyClip.app dist
```

Skip Apple with `RALLYCLIP_SKIP_SIGNING=1` or `RALLYCLIP_SKIP_NOTARIZE=1`.

The runtime is torch-free: pose inference runs on the ONNX bundled in
`models/rallyclip_v0.5.0/` via onnxruntime (`extraction/yolo_onnx_runner.py`).
Training and the legacy .pt path need `pip install ".[train]"`.

### Headless mode
The shipped binary can run the full pipeline without launching the GUI. Pass
`--cli` as the first argument; everything after it is the regular `rallyclip`
CLI:

```bash
dist/RallyClip.app/Contents/MacOS/RallyClip --cli --video match.mp4 \
  --start-time 1240 --duration 180 \
  --write-csv --csv-output-dir /tmp/out --no-segment-video
```

`RallyClip --cli --help` prints the full flag reference. Pose ONNX weights ship
inside the app bundle (`models/rallyclip_v0.5.0/`); no extra download is needed.
