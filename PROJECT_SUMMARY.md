# Project Summary

## What RallyClip Is
RallyClip is the public open-source repo for:

1. the local `rallyclip` CLI
2. the local inference/runtime package
3. local GUI code
4. the training pipeline code

The hosted production product is not part of this repository. It now lives in the private `rallyclip-prod` repo.

## Product Goal
RallyClip exists to extract only the actual points from full tennis match footage.

The core workflow is:
- take a full match video
- remove the dead time between points
- return a condensed video containing only the action
- optionally return structured timestamps/CSV output

The public repo is meant to support that workflow locally on a user's own machine.

## Current Inference Stack
Inference is artifact-driven rather than loose-checkpoint driven.

Current packaged artifact:
- `models/rallyclip_v0.3.1/model.onnx`
- `models/rallyclip_v0.3.1/scaler.json`
- `models/rallyclip_v0.3.1/manifest.json`

The manifest is the source of truth for:
- temporal inference settings
- feature/scaler metadata
- postprocessing defaults

The current default artifact uses:
- YOLO pose extraction at `5 fps`
- sequence length `100` frames (`20s`)
- overlap `50` frames
- hysteresis postprocessing:
  - `low=0.45`
  - `high=0.7`
  - `sigma=1.0`
  - `min_dur_sec=1.0`

## CLI
The CLI command is `rallyclip`.

Typical flow:
1. Run YOLO pose extraction on a match video.
2. Build features for each frame window.
3. Scale features using `scaler.json`.
4. Run ONNX inference with the packaged artifact.
5. Apply postprocessing and write segmented output video and optional CSV.

Key outputs:
- segmented video in `output_videos/`
- optional CSV in `output_csvs/`

Main config file:
- `config.toml`

## Training Pipeline
The training system is a separate step-based pipeline driven by:
- `configs/train/base.yaml`
- `train.py`

Main pipeline stages:
1. extract
2. preprocess
3. features
4. dataset
5. train
6. eval

Training docs:
- `docs/training.md`

The training code is public. Training data, private evaluation sets, annotations, and any paid/private infrastructure are kept out of this repository.

## Repo Orientation
Important paths:
- `src/` — Python package code
- `tests/` — pytest coverage
- `models/` — packaged inference artifacts and weights
- `configs/` — training/runtime config
- `docs/` — training docs
- `gui/frontend/` — local GUI assets

## Near-Term Priorities
1. Keep the CLI stable as the primary public product.
2. Improve local inference quality and postprocessing.
3. Keep training code open while keeping training data private.
4. Continue local GUI work without coupling this repo to hosted production infrastructure.
