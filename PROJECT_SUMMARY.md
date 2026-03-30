# Project Summary

## What RallyClip Is
RallyClip is a tennis match segmentation project with three main surfaces:

1. A local Python CLI for running point segmentation on full match videos.
2. A hosted web app at `https://rallyclip.vercel.app`.
3. A training pipeline for producing new model artifacts.

The primary user-facing product right now is the CLI.

## What The Project Should Be
RallyClip should be a free, open-source way to extract only the actual points from full tennis match recordings.

The product goal is:
- take a full match video
- remove the dead time between points
- return a condensed video containing only the action
- optionally return structured timestamps/CSV output

The broader product direction from the older project docs is:
- free match segmentation should be available without a paid subscription
- the default experience should work locally on a user's own machine
- the system should stay computationally efficient enough for laptop use
- over time, the product can support both local use and a hosted/cloud option

So the intended shape of the project is not just "a model repo." It is:
- a usable local tool for real players reviewing their matches
- a clean packaged inference artifact and CLI
- a hosted frontend surface for account/product workflows
- an evolving training pipeline that keeps improving the segmentation quality

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

## Web App
The current hosted app is:
- `https://rallyclip.vercel.app`

The active Next.js web app lives under:
- `app/web/`

Current web scope is mostly account/product shell work:
- auth
- onboarding/profile flows
- account management

It is not the primary documented runtime path for inference yet.

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

Training artifacts are written under:
- `data/datasets/<run_id>/`
- `data/runs/<run_id>/`

## Current Model Direction
The currently locked classical recipe is:
- YOLO pose features
- sequence model over engineered features
- mirrored training augmentation
- tuned hysteresis postprocessing

Recent deployment work standardized model export into versioned artifact directories containing:
- `model.onnx`
- `scaler.json`
- `manifest.json`

## Repo Orientation
Important paths:
- `src/` — Python package code
- `tests/` — pytest coverage
- `app/web/` — hosted Next.js app
- `models/` — packaged inference artifacts and weights
- `configs/` — training/runtime config
- `docs/` — training docs and older static docs assets

## Near-Term Priorities
1. Keep the CLI stable as the primary local product.
2. Continue deploying the hosted web app separately on Vercel.
3. Improve postprocessing and uncertainty/review scoring.
4. Scale training data and continue model iteration.
