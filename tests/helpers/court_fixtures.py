"""Shared helpers for the court-detection regression flights.

Both flights run over the same hand-annotated set, frozen under
``tests/fixtures/court/`` (clean frames + golden out-masks + manifest). The
deterministic flight feeds the frozen frames straight into detection; the e2e
flight runs the live pipeline (YOLO + RANSAC homography) on the source videos.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "court"

# The e2e flight's source videos / weights are machine-specific and live outside the repo,
# so they come only from env. When unset the resolvers return None and the flight skips with
# a self-explanatory "set RALLYCLIP_..." reason (no personal paths committed to the repo).


def load_manifest() -> list[dict]:
    """Detected fixtures only (those with a frozen frame + golden mask)."""
    data = json.loads((FIX_DIR / "manifest.json").read_text())
    return [m for m in data if m.get("detected")]


def fixture_ids() -> list[str]:
    return [m["id"] for m in load_manifest()]


def manifest_by_id() -> dict[str, dict]:
    return {m["id"]: m for m in load_manifest()}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a) > 127
    b = np.asarray(b) > 127
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum()) / float(union)


def resolve_video_dir() -> Path | None:
    raw = os.environ.get("RALLYCLIP_COURT_VIDEO_DIR")
    p = Path(raw) if raw else None
    return p if p and p.is_dir() else None


def resolve_yolo_weights() -> Path | None:
    raw = os.environ.get("RALLYCLIP_YOLO_WEIGHTS")
    p = Path(raw) if raw else None
    return p if p and p.is_file() else None
