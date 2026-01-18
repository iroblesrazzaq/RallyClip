from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_STEPS = ["extract", "preprocess", "features", "dataset", "train", "eval"]


def run_pipeline(config: Dict[str, Any], steps_override: Optional[Iterable[str]] = None) -> None:
    steps = list(steps_override) if steps_override is not None else list(config.get("steps", DEFAULT_STEPS))
    run_id = config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_id"] = run_id

    data_root = Path(config.get("data_root", "data")).expanduser().resolve()
    config["data_root"] = str(data_root)

    logger.info("Run id: %s", run_id)
    logger.info("Steps: %s", ", ".join(steps))

    step_map = {
        "extract": _run_extract,
        "preprocess": _run_preprocess,
        "features": _run_features,
        "dataset": _run_dataset,
        "train": _run_train,
        "eval": _run_eval,
    }

    for step in steps:
        func = step_map.get(step)
        if func is None:
            raise ValueError(f"Unknown step: {step}")
        func(config)


def _run_extract(config: Dict[str, Any]) -> None:
    logger.info("TODO: extract YOLO outputs (native fps)")


def _run_preprocess(config: Dict[str, Any]) -> None:
    logger.info("TODO: preprocess/downsample and apply court filtering")


def _run_features(config: Dict[str, Any]) -> None:
    logger.info("TODO: feature engineering")


def _run_dataset(config: Dict[str, Any]) -> None:
    logger.info("TODO: dataset creation")


def _run_train(config: Dict[str, Any]) -> None:
    logger.info("TODO: training loop")


def _run_eval(config: Dict[str, Any]) -> None:
    logger.info("TODO: evaluation")
