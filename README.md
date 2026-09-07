# RallyClip

RallyClip is an open-source tool for tennis video segmentation. It extracts rally/point intervals from full match footage and outputs a segmented video plus optional CSV timestamps.

This repo ships:
- `rallyclip` CLI for local inference
- `rallyclip-desktop` for macos desktop app
- `rallyclip gui` browser-based local UI for development
- open training pipeline code

## To run

**macOS with an M-series chip (M1, M2, M3, …)**

Install the desktop app from [GitHub Releases](https://github.com/iroblesrazzaq/RallyClip/releases/latest).

Open the DMG and drag RallyClip into Applications. Apple Silicon only.

**Windows, Linux, and Intel Macs**

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it (`brew install uv`, or the installer on that page). uv will create a local environment and can install Python 3.10+.

```bash
git clone --depth 1 --single-branch https://github.com/iroblesrazzaq/RallyClip.git
cd RallyClip
uv sync --extra cpu
uv run python scripts/fetch_artifact.py
uv run rallyclip gui
```

That starts a local backend and opens RallyClip in your browser at `http://127.0.0.1:8000` (or the next free port). Leave the terminal open while you use it.

NVIDIA GPU with CUDA:

```bash
uv sync --extra gpu
uv run rallyclip gui
```

Do not combine the `cpu` and `gpu` extras — they install conflicting ONNX Runtime wheels.

CLI, training, and extra install notes live in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Features coming soon (in rough order)
- iOS mobile app beta coming soon

## Current release status
The current public desktop release is `v0.3.0` for **Apple Silicon macOS only**.

## Features down the road
In no particular order,
- OPTIMIZE TF OUTTA THE MODELS TO MAKE THEM FASTER AND BETTER!!!!
- Mobile app (once I can scale data more to push down model size by expanding the repertoire of architectures I can use, particularly in training more complex deep learning models from scratch)
- Doubles support (need to label + train on doubles data)
- Open dataset (once deployed, opt-in for publicly available dataset for open-source community to use)
- Match scoring
- Better resolution support: 720p, 1080p, 1440p, 4K. Training models for each, finding out best yolo params etc for them, given that YOLO downscales regardless it might not matter, but will look into.
