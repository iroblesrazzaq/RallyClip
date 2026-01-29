from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from preprocessing.court_detector_impl import CourtDetector
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


class CourtMaskCache:
    def __init__(self, model_path: str = "yolov8s.pt", target_time: int = 60) -> None:
        self.model_path = model_path
        self.target_time = target_time

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
        detector = CourtDetector(yolo_model_path=self.model_path)
        mask, clean_frame, metadata = detector.process_video(str(video_path), target_time=self.target_time)

        self._inject_line_metadata(detector, metadata, clean_frame)
        lines = self._extract_lines(metadata)
        success = bool(metadata.get("court_detection_success")) and mask is not None

        meta = {
            "video": _video_metadata(video_path),
            "detector": {
                "model_path": self.model_path,
                "target_time": self.target_time,
            },
            "heuristics": self._extract_heuristics(metadata, clean_frame),
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        if not success:
            mask = None

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
            return cached

        result = self.compute(video_path)
        self.save(cache_path, result)
        return result

    def _extract_lines(self, metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        baseline = metadata.get("baseline")
        left_line = metadata.get("left_doubles_sideline")
        right_line = metadata.get("right_doubles_sideline")
        extended = metadata.get("extended_sidelines")

        lines = []
        for entry in (baseline, left_line, right_line):
            if entry is not None:
                lines.extend(entry)
        if extended:
            lines.extend(extended)
        return _safe_cast_lines(lines)

    def _extract_heuristics(self, metadata: Dict[str, Any], clean_frame: np.ndarray) -> Dict[str, Any]:
        heuristics: Dict[str, Any] = {}
        image_width = metadata.get("image_width") or (clean_frame.shape[1] if clean_frame is not None else None)
        if image_width:
            baseline_width = metadata.get("baseline_width") or 0
            heuristics["baseline_width_pct"] = float((baseline_width / image_width) * 100)
            heuristics["full_width_baseline"] = heuristics["baseline_width_pct"] > 98.5
        return heuristics

    def _inject_line_metadata(self, detector: CourtDetector, metadata: Dict[str, Any], clean_frame: np.ndarray) -> None:
        if clean_frame is None:
            return
        try:
            horizontal, vertical, right_diag, left_diag = detector.detect_court_lines(clean_frame)
            merged_horizontal = detector.merge_lines(horizontal, clean_frame.shape, kernel_size=(5, 30))
            merged_right = detector.merge_lines(right_diag, clean_frame.shape, kernel_size=(2, 2))
            merged_left = detector.merge_lines(left_diag, clean_frame.shape, kernel_size=(2, 2))
            baseline = detector.find_baseline(merged_horizontal)
            left_sideline = detector.process_side_decision_tree(merged_left, baseline, clean_frame.shape[1], "left")
            right_sideline = detector.process_side_decision_tree(merged_right, baseline, clean_frame.shape[1], "right")
            metadata.setdefault("baseline", baseline)
            metadata.setdefault("left_doubles_sideline", left_sideline)
            metadata.setdefault("right_doubles_sideline", right_sideline)
            metadata.setdefault("extended_sidelines", self._compute_extended_lines(clean_frame, baseline, left_sideline, right_sideline))
        except Exception as exc:
            logger.debug("Failed to inject line metadata: %s", exc)

    @staticmethod
    def _compute_extended_lines(
        clean_frame: np.ndarray,
        baseline: Optional[list],
        left_sideline: Optional[list],
        right_sideline: Optional[list],
    ) -> Optional[list]:
        if baseline is None or left_sideline is None or right_sideline is None:
            return None

        BASE_HORIZONTAL_SHIFT = 100
        screen_width = clean_frame.shape[1]
        bx1, by1, bx2, by2 = baseline[0]
        baseline_width = abs(bx2 - bx1)
        scale_factor = (baseline_width / screen_width) if screen_width else 0.0
        dynamic_shift = BASE_HORIZONTAL_SHIFT * scale_factor

        lx1, ly1, lx2, ly2 = left_sideline[0]
        left_slope = (ly2 - ly1) / (lx2 - lx1) if (lx2 - lx1) != 0 else float("inf")
        left_intercept = ly1 - left_slope * lx1

        rx1, ry1, rx2, ry2 = right_sideline[0]
        right_slope = (ry2 - ry1) / (rx2 - rx1) if (rx2 - rx1) != 0 else float("inf")
        right_intercept = ry1 - right_slope * rx1

        if left_slope != float("inf"):
            left_shifted_intercept = left_intercept - dynamic_shift / np.sqrt(1 + left_slope**2)
            left_extended = [
                0,
                int(left_slope * 0 + left_shifted_intercept),
                screen_width,
                int(left_slope * screen_width + left_shifted_intercept),
            ]
        else:
            left_shifted_x = int(lx1 - dynamic_shift)
            left_extended = [left_shifted_x, 0, left_shifted_x, clean_frame.shape[0]]

        if right_slope != float("inf"):
            right_shifted_intercept = right_intercept - dynamic_shift / np.sqrt(1 + right_slope**2)
            right_extended = [
                0,
                int(right_slope * 0 + right_shifted_intercept),
                screen_width,
                int(right_slope * screen_width + right_shifted_intercept),
            ]
        else:
            right_shifted_x = int(rx1 + dynamic_shift)
            right_extended = [right_shifted_x, 0, right_shifted_x, clean_frame.shape[0]]

        return [left_extended, right_extended]
