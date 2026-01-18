from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)


def render_stage(stage: str, config: Dict[str, Any], videos: Iterable[str]) -> None:
    stage_map = {
        "yolo": _render_yolo,
        "court": _render_court,
        "preproc": _render_preproc,
    }
    if stage not in stage_map:
        raise ValueError(f"Unknown stage: {stage}")

    for video in videos:
        stage_map[stage](video, config)


def _render_yolo(video: str, config: Dict[str, Any]) -> None:
    logger.info("TODO: render YOLO visualization for %s", video)


def _render_court(video: str, config: Dict[str, Any]) -> None:
    logger.info("TODO: render court visualization for %s", video)


def _render_preproc(video: str, config: Dict[str, Any]) -> None:
    logger.info("TODO: render preproc visualization for %s", video)
