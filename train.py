#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from training.io.config import load_config  # noqa: E402
from training.io.videos import create_flipped_videos, resolve_videos  # noqa: E402
from training.paths import annotations_dir, raw_videos_dir, resolve_data_root  # noqa: E402
from training.pipeline import export_model_artifact, run_pipeline, run_postprocess_sweep, run_sweep  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RallyClip training pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--steps",
        help="Comma-separated step list to override config (extract,preprocess,features,dataset,train,eval)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run configured sweep loops from config.sweep",
    )
    parser.add_argument(
        "--postprocess-sweep",
        action="store_true",
        help="Evaluate default post-processing over completed sweep runs",
    )
    parser.add_argument(
        "--make-flipped-videos",
        action="store_true",
        help="Create horizontally flipped training-video variants under a separate raw-video root",
    )
    parser.add_argument(
        "--export-artifact",
        action="store_true",
        help="Export a finished training run into a versioned deployment artifact directory",
    )
    parser.add_argument("--export-run-id", help="Run id to export when using --export-artifact")
    parser.add_argument("--export-version", help="Versioned artifact directory name when using --export-artifact")
    parser.add_argument(
        "--export-checkpoint",
        default="best",
        choices=["best", "last"],
        help="Checkpoint name to export when using --export-artifact",
    )
    parser.add_argument(
        "--export-output-dir",
        help="Optional output root for exported artifacts (default: ./models)",
    )
    parser.add_argument(
        "--export-overwrite",
        action="store_true",
        help="Overwrite an existing exported artifact directory when using --export-artifact",
    )
    parser.add_argument(
        "--flip-mode",
        default="annotated",
        choices=["annotated", "all", "list"],
        help="Which videos to flip when using --make-flipped-videos",
    )
    parser.add_argument(
        "--flip-output-dir",
        help="Optional output root for flipped videos (default: <data_root>/raw_videos_flip_h)",
    )
    parser.add_argument(
        "--flip-overwrite",
        action="store_true",
        help="Overwrite existing flipped videos when using --make-flipped-videos",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg binary to use for --make-flipped-videos",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    steps_override = [s.strip() for s in args.steps.split(",") if s.strip()] if args.steps else None

    selected_modes = [
        bool(args.sweep),
        bool(args.postprocess_sweep),
        bool(args.make_flipped_videos),
        bool(args.export_artifact),
    ]
    if sum(1 for mode in selected_modes if mode) > 1:
        raise SystemExit("--sweep, --postprocess-sweep, --make-flipped-videos, and --export-artifact cannot be used together")

    if args.sweep:
        if steps_override is not None:
            raise SystemExit("--steps cannot be used with --sweep")
        run_sweep(config)
        return 0

    if args.postprocess_sweep:
        if steps_override is not None:
            raise SystemExit("--steps cannot be used with --postprocess-sweep")
        run_postprocess_sweep(config)
        return 0

    if args.make_flipped_videos:
        if steps_override is not None:
            raise SystemExit("--steps cannot be used with --make-flipped-videos")
        data_root = resolve_data_root(config)
        raw_dir = raw_videos_dir(data_root)
        ann_dir = annotations_dir(data_root)
        videos = resolve_videos(args.flip_mode, raw_dir, ann_dir, config.get("videos"))
        output_dir = Path(args.flip_output_dir).expanduser().resolve() if args.flip_output_dir else (data_root / "raw_videos_flip_h")
        created = create_flipped_videos(
            raw_dir=raw_dir,
            output_dir=output_dir,
            videos=videos,
            ffmpeg_bin=args.ffmpeg_bin,
            overwrite=args.flip_overwrite,
        )
        logging.info("Prepared %d flipped videos under %s", len(created), output_dir)
        return 0

    if args.export_artifact:
        if steps_override is not None:
            raise SystemExit("--steps cannot be used with --export-artifact")
        if not args.export_run_id:
            raise SystemExit("--export-run-id is required with --export-artifact")
        if not args.export_version:
            raise SystemExit("--export-version is required with --export-artifact")
        data_root = resolve_data_root(config)
        artifact_dir = export_model_artifact(
            data_root=data_root,
            run_id=args.export_run_id,
            version=args.export_version,
            checkpoint_name=args.export_checkpoint,
            output_root=args.export_output_dir,
            overwrite=args.export_overwrite,
        )
        logging.info("Exported artifact to %s", artifact_dir)
        return 0

    run_pipeline(config, steps_override=steps_override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
