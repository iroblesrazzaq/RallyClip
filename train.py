#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from training.io.config import load_config  # noqa: E402
from training.pipeline import run_pipeline, run_postprocess_sweep, run_sweep  # noqa: E402


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    steps_override = [s.strip() for s in args.steps.split(",") if s.strip()] if args.steps else None

    if args.sweep and args.postprocess_sweep:
        raise SystemExit("--sweep and --postprocess-sweep cannot be used together")

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

    run_pipeline(config, steps_override=steps_override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
