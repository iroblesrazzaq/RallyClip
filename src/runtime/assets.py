"""Shared asset/manifest resolution used by the CLI, GUI, and desktop bundle."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

YOLO_SIZE_MAP = {
    "nano": "yolov8n-pose.pt",
    "small": "yolov8s-pose.pt",
    "medium": "yolov8m-pose.pt",
    "large": "yolov8l-pose.pt",
}


def candidate_roots() -> list[Path]:
    """Possible roots where assets might live (frozen bundle, repo root, cwd, site-packages)."""
    here = Path(__file__).resolve()
    roots = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass).resolve())
    roots.append(Path.cwd())
    for depth in (2, 3, 4):
        try:
            roots.append(here.parents[depth])
        except IndexError:
            continue
    seen: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.append(r)
    return seen


def resolve_asset(explicit: Optional[str], env_var: str, relatives: list[str], description: str) -> Path:
    """Resolve a required asset from CLI/config/env/default locations."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at '{path}'")
        return path

    env_val = os.environ.get(env_var)
    if env_val:
        path = Path(env_val).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at '{path}' (from {env_var})")
        return path

    for root in candidate_roots():
        for rel in relatives:
            candidate = (Path(root) / rel).expanduser()
            if candidate.exists():
                return candidate.resolve()

    roots_str = ", ".join(str(r) for r in candidate_roots())
    raise FileNotFoundError(
        f"{description} not found. Set via CLI flag, config, or env {env_var}; "
        f"searched relative locations {relatives} under: {roots_str}"
    )


def manifest_values(model_path: Path, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the model's manifest into a flat dict of contract + postprocess values.

    Returns {} if no manifest can be found/parsed; callers crash on missing required
    fields rather than substituting phantom defaults.
    """
    manifest_path = manifest_path or (model_path.parent / "manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning("Found manifest at %s but could not parse it (%s); ignoring it.", manifest_path, e)
        return {}

    inference = payload.get("inference", {}) or {}
    postprocess = (payload.get("postprocess", {}) or {}).get("params", {}) or {}
    feature_pipeline = payload.get("feature_pipeline", {}) or {}
    values: Dict[str, Any] = {}

    # Contract (immutable)
    if feature_pipeline.get("target_fps") is not None:
        values["fps"] = float(feature_pipeline["target_fps"])
    if inference.get("seq_len_frames") is not None:
        values["seq_len"] = int(inference["seq_len_frames"])
    if feature_pipeline.get("imgsz") is not None:
        values["imgsz"] = int(float(feature_pipeline["imgsz"]))
    if feature_pipeline.get("conf") is not None:
        values["conf"] = float(feature_pipeline["conf"])
    if feature_pipeline.get("feature_set") is not None:
        values["feature_set"] = str(feature_pipeline["feature_set"])
    if feature_pipeline.get("screen_width") is not None:
        values["screen_width"] = int(feature_pipeline["screen_width"])
    if feature_pipeline.get("screen_height") is not None:
        values["screen_height"] = int(feature_pipeline["screen_height"])
    if feature_pipeline.get("yolo_model") is not None:
        values["yolo_model"] = str(feature_pipeline["yolo_model"])

    # Postprocess (mutable)
    if inference.get("overlap_frames") is not None:
        values["overlap"] = int(inference["overlap_frames"])
    if postprocess.get("sigma") is not None:
        values["sigma"] = float(postprocess["sigma"])
    if postprocess.get("low") is not None:
        values["low"] = float(postprocess["low"])
    if postprocess.get("high") is not None:
        values["high"] = float(postprocess["high"])
    if postprocess.get("min_dur_sec") is not None:
        values["min_dur_sec"] = float(postprocess["min_dur_sec"])
    return values
