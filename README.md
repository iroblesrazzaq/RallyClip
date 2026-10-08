# RallyClip

RallyClip is an open-source tool for tennis video segmentation. It extracts rally/point intervals from full match footage and outputs a segmented video plus optional CSV timestamps.

The new macOS desktop app uses SwiftUI and AVFoundation for playback, editing,
and export. It bundles the v0.6 ONNX analysis engine, including the learned court
and net detector and separate near/far player features. Everything runs locally.
The original RallyClip logo stays; the interface uses native controls, the system
font, and the system light/dark appearance.

Native build, development, testing, and signing: [docs/native-macos.md](docs/native-macos.md).

## To run

**macOS with an M-series chip (M1, M2, M3, …)**

Install the desktop app from [GitHub Releases](https://github.com/iroblesrazzaq/RallyClip/releases).

Open the DMG and drag RallyClip into Applications. The native app requires macOS
14 or newer and Apple Silicon. Choose a video or a folder of consecutive MP4
clips; review points, save boundary edits, then export a video, selected
highlights, separate point videos, or CSV timestamps. Keep original footage
available for playback and full-quality export.

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

For camera-split matches, choose **New Match → Choose a folder of consecutive
clips**. RallyClip naturally orders the MP4 files, analyses a 720p proxy as one
continuous timeline, and exports the detected points from the original footage
at its original resolution. Keep the source folder available until export.

NVIDIA GPU with CUDA:

```bash
uv sync --extra gpu
uv run rallyclip gui
```

Do not combine the `cpu` and `gpu` extras — they install conflicting ONNX Runtime wheels.

CLI, training, and extra install notes live in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Features coming soon (in rough order)
- iOS mobile app beta coming soon
- doubles support
- further model performance and inference speed improvements


## Current release status
The current public stable desktop release is `v0.5.0`. The native v0.6 desktop
is being prepared on `feat/native-macos`; this branch's release workflow builds,
signs, and notarizes it using the existing Apple credentials. The verified v0.6
[model artifact](https://github.com/iroblesrazzaq/RallyClip/releases/tag/artifact-rallyclip_v0.6.0)
is available for source and CI builds.

## Stuff I'm thinking about
Current non-user-facing features and long term ideas
- OPTIMIZE TF OUTTA THE MODELS TO MAKE THEM FASTER AND BETTER!!!! (always)
- The v0.6 model now uses learned court/net geometry and two pose passes for near/far players. More training data and better court robustness remain priorities.
- Match scoring: seems pretty hard. Will need a rework of architecture to track way more stuff (need player identity, tracking who won point, which will require much more advanced arch and more data presumably)

The next platform is iOS. The native desktop's media and point-timeline code is
isolated in `RallyClipKit`, while inference still uses the proven bundled Python
engine on macOS.
