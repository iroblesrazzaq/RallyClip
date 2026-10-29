from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from preprocessing.court_lines import detect_court, padded_sidelines
from training.paths import pose_courts_dir

logger = logging.getLogger(__name__)


@dataclass
class CourtCacheResult:
    success: bool
    mask: Optional[np.ndarray]
    metadata: Dict[str, Any]
    lines: Optional[np.ndarray]


def _video_metadata(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = float(frames / fps) if fps else None
    return {
        "width": width,
        "height": height,
        "fps": float(fps),
        "frames": frames,
        "duration": duration,
    }


def _serialize_metadata(metadata: Dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True)


def _safe_cast_lines(lines: Optional[list]) -> Optional[np.ndarray]:
    if lines is None:
        return None
    try:
        return np.asarray(lines, dtype=np.float32)
    except Exception:
        return None


def _warn_on_video_metadata_mismatch(cache_meta: Dict[str, Any], current_meta: Dict[str, Any]) -> None:
    for key in ("width", "height", "fps"):
        if key in cache_meta and key in current_meta and cache_meta[key] != current_meta[key]:
            logger.warning("Court cache metadata mismatch for %s: %s != %s", key, cache_meta[key], current_meta[key])


# Recorded in each cache's meta; a cache from another detector is recomputed.
DETECTOR_KIND = "court_lines"


def _segment(a, b) -> List[List[float]]:
    return [[float(a[0]), float(a[1]), float(b[0]), float(b[1])]]


def _court_lines(points: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    """Doubles sidelines, near baseline and padded sidelines, in the [[x1, y1, x2, y2]]
    layout training.viz reads. Empty when a corner is missing."""
    lines = padded_sidelines(points, width)
    if lines is None:
        return {}
    nl, fl, nr, fr = (points[n] for n in ("near_left", "far_left", "near_right", "far_right"))
    extended = [
        [float(value), 0.0, float(value), float(height)] if slope is None
        else [0.0, float(value), float(width), float(slope * width + value)]
        for slope, value in lines
    ]
    return {
        "baseline": _segment(nl, nr),
        "left_doubles_sideline": _segment(nl, fl),
        "right_doubles_sideline": _segment(nr, fr),
        "extended_sidelines": extended,
    }


class CourtMaskCache:
    """Per-video court 'out' mask from the court line model (court_lines.onnx)."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def cache_path(self, data_root: Path, video_path: Path) -> Path:
        return pose_courts_dir(data_root) / f"{video_path.stem}.npz"

    def load(self, cache_path: Path, current_video_meta: Optional[Dict[str, Any]] = None) -> Optional[CourtCacheResult]:
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path, allow_pickle=True)
            metadata_json = data.get("metadata_json")
            metadata = json.loads(metadata_json.item()) if metadata_json is not None else {}
            if current_video_meta:
                _warn_on_video_metadata_mismatch(metadata.get("video", {}), current_video_meta)
            mask = data.get("mask") if data.get("success") else None
            lines = data.get("lines")
            success = bool(data.get("success"))
            return CourtCacheResult(success=success, mask=mask, metadata=metadata, lines=lines)
        except Exception as exc:
            logger.warning("Failed to load court cache %s: %s", cache_path, exc)
            return None

    def compute(self, video_path: Path) -> CourtCacheResult:
        court = detect_court(str(video_path), self.model_path)
        mask = court.out_mask() if court is not None else None
        geometry = court.geometry() if court is not None else {}
        metadata: Dict[str, Any] = {
            "court_detection_success": mask is not None,
            "error": None if mask is not None else "court model gave no mask",
            "points_px": geometry.get("points_px"),
            "lines_ok": geometry.get("lines_ok"),
        }
        if court is not None:
            metadata.update(_court_lines(geometry["points_px"], *court.frame_size))
        lines = self._extract_lines(metadata)
        success = mask is not None
        if not success:
            logger.warning("COURT DETECTION FAILED for %s -> mask unavailable", video_path.name)

        meta = {
            "video": _video_metadata(video_path),
            "detector": {"kind": DETECTOR_KIND, "model_path": self.model_path},
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        return CourtCacheResult(success=success, mask=mask, metadata=meta, lines=lines)

    def save(self, cache_path: Path, result: CourtCacheResult) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "success": np.array(bool(result.success)),
            "metadata_json": np.array(_serialize_metadata(result.metadata)),
        }
        if result.mask is not None:
            payload["mask"] = result.mask.astype(np.uint8)
        if result.lines is not None:
            payload["lines"] = result.lines.astype(np.float32)
        np.savez_compressed(cache_path, **payload)

    def get_or_create(
        self,
        data_root: Path,
        video_path: Path,
        force: bool = False,
    ) -> CourtCacheResult:
        current_meta = _video_metadata(video_path)
        cache_path = self.cache_path(data_root, video_path)

        cached = self.load(cache_path, current_video_meta=current_meta)
        if cached and not force:
            if cached.metadata.get("detector", {}).get("kind") == DETECTOR_KIND:
                return cached
            logger.info("Court cache %s is from the classic detector; recomputing", cache_path.name)

        result = self.compute(video_path)
        self.save(cache_path, result)
        return result

    def _extract_lines(self, metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        lines = []
        for key in ("baseline", "left_doubles_sideline", "right_doubles_sideline"):
            if metadata.get(key) is not None:
                lines.extend(metadata[key])
        if metadata.get("extended_sidelines"):
            lines.extend(metadata["extended_sidelines"])
        return _safe_cast_lines(lines) if lines else None
