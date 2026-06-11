from __future__ import annotations

import json
import os
from pathlib import Path

FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "quality"


def load_manifest() -> list[dict]:
    return json.loads((FIX_DIR / "manifest.json").read_text(encoding="utf-8"))


def resolve_video_dir() -> Path | None:
    raw = os.environ.get("RALLYCLIP_EVAL_VIDEO_DIR")
    path = Path(raw).expanduser() if raw else None
    return path if path and path.is_dir() else None


def resolve_release_bin() -> Path | None:
    raw = os.environ.get("RALLYCLIP_RELEASE_BIN")
    path = Path(raw).expanduser() if raw else None
    return path if path and path.is_file() else None


def resolve_artifact_dir() -> Path | None:
    raw = os.environ.get("RALLYCLIP_EVAL_ARTIFACT_DIR")
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"RALLYCLIP_EVAL_ARTIFACT_DIR is not a directory: {path}")
    return path
