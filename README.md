# RallyClip

RallyClip is an open-source tool for tennis video segmentation. It extracts rally/point intervals from full match footage and outputs a segmented video plus optional CSV timestamps.

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
- improved court detector + model performance:
    - using a pretrained HRNet base (to maintain high resolution court line features)
    - objective: model outputs 5 heatmaps of 128x72 pixels, each with a line (near and far baselines, doubles sidelines, and the net line (bottom of the net).
          - the ground truth is the actual line segment where each target is exp(-d^2/2), where d is the distance from a given pixel to the nearest point on the line segment. This means that on the line segment, the target is about 1, and it goes to zero pretty fast as we move away from the line. Then, we do BCE over each pixel in our 5 output maps, weighting line pixels more heavily.
    - the super cool part is that I had opus 5.5 label a bunch of court images and it was scarily accurate (very few cases where human could do better). So I could automate my data labelling :). 
- iOS mobile app beta coming soon
- doubles support



## Current release status
The current public desktop release is `v0.5.0` for **Apple Silicon macOS only**.

## Stuff I'm thinking about
Current non-user-facing features and long term ideas
- OPTIMIZE TF OUTTA THE MODELS TO MAKE THEM FASTER AND BETTER!!!! (always)
- Rework of court annotations: switching from heuristics to DL models, bootstrapping with my heuristic annotations. [IN PROGRESS: RESULTS ARE +6% PERF :)]
    - current court mask pipeline fails in some edge cases
    - to improve the model's vision for the far court player(s), we need 2 YOLO passes, one over a zoomed in section and the other one over the whole court. However, this runs into the issue of redundant information: ex the near court player showing up on the zoomed in YOLO. My current goal is to have net annotations so if a player's bounding box goes below the bottom of the net, that identifies them as the near court player and we ignore. This is more hand-crafted and less bitter-lessoney but in a low-data regime, I think its the best step moving forward. Also should be fun training a new type of model
    - using a pretrained HRNet base (to maintain high resolution court line features)
    - objective: model outputs 5 heatmaps of 128x72 pixels, each with a line (near and far baselines, doubles sidelines, and the net line (bottom of the net).
          - the ground truth is the actual line segment where each target is exp(-d^2/2), where d is the distance from a given pixel to the nearest point on the line segment. This means that on the line segment, the target is about 1, and it goes to zero pretty fast as we move away from the line. Then, we do BCE over each pixel in our 5 output maps, weighting line pixels more heavily.
    - the super cool part is that I had opus 5.5 label a bunch of court images and it was scarily accurate (very few cases where human could do better). So I could automate my data labelling :). 
 
- Match scoring: seems pretty hard. Will need a rework of architecture to track way more stuff (need player identity, tracking who won point, which will require much more advanced arch and more data presumably)
- 4k support
- allow for files > 2Gb on desktop (web gui with local backend is fine) cuz i fixed memory issue so now we stream and dont load everything into memory lol

My next big push with this project will be training this court model (yay fun) and iOS support (although i have neither knowledge nor passion for mobile dev but let the tokens flow...). 
