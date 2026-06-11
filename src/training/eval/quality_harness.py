from __future__ import annotations

import csv
import json
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from training.metrics.segment import (
    compute_time_point_classification_metrics,
    compute_time_segment_metrics,
)

DEFAULT_FPS = 5.0
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_WELL_COVERAGE_THRESHOLD = 0.9


@dataclass(frozen=True)
class QualityEntry:
    id: str
    video: str
    gt: str
    start_time_s: int
    duration_s: int
    gt_points_in_window: int


@dataclass(frozen=True)
class RunnerConfig:
    mode: str = "dev"
    release_bin: Path | None = None
    command: tuple[str, ...] = (sys.executable, "-m", "cli.main")
    extra_args: tuple[str, ...] = ()


def load_manifest(path: Path) -> list[QualityEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [QualityEntry(**entry) for entry in data]


def run_quality_eval(
    *,
    manifest_path: Path,
    video_dir: Path,
    gt_dir: Path,
    output_path: Path | None = None,
    runner: RunnerConfig | None = None,
    fps: float = DEFAULT_FPS,
) -> dict[str, Any]:
    entries = load_manifest(manifest_path)
    runner = runner or RunnerConfig()

    reports = [
        evaluate_entry(
            entry,
            video_dir=video_dir,
            gt_dir=gt_dir,
            runner=runner,
            fps=fps,
        )
        for entry in entries
    ]
    report = {
        "runner": _runner_label(runner),
        "manifest": _path_label(manifest_path),
        "video_dir": _path_label(video_dir),
        "gt_dir": _path_label(gt_dir),
        "fps": fps,
        "per_video": reports,
        "pooled": _pool_reports(reports),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def evaluate_entry(
    entry: QualityEntry,
    *,
    video_dir: Path,
    gt_dir: Path,
    runner: RunnerConfig,
    fps: float = DEFAULT_FPS,
) -> dict[str, Any]:
    video_path = video_dir / entry.video
    gt_path = gt_dir / entry.gt
    if not video_path.is_file():
        raise FileNotFoundError(f"{entry.id}: source video missing: {video_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"{entry.id}: ground-truth annotation missing: {gt_path}")

    true_segments = load_windowed_gt_segments(
        gt_path,
        start_time_s=float(entry.start_time_s),
        duration_s=float(entry.duration_s),
    )
    if len(true_segments) != int(entry.gt_points_in_window):
        raise ValueError(
            f"{entry.id}: expected {entry.gt_points_in_window} GT points after windowing, "
            f"got {len(true_segments)}"
        )

    with tempfile.TemporaryDirectory(prefix=f"rallyclip_quality_{entry.id}_") as tmp:
        tmp_path = Path(tmp)
        csv_path = run_pipeline_to_csv(
            video_path=video_path,
            entry=entry,
            csv_output_dir=tmp_path,
            runner=runner,
        )
        pred_segments = load_segments_csv(csv_path)

    timestamps = _timestamps(float(entry.duration_s), float(fps))
    true_binary = intervals_to_binary(true_segments, timestamps)
    pred_binary = intervals_to_binary(pred_segments, timestamps)
    segment_metrics = compute_time_segment_metrics(true_binary, pred_binary, timestamps)
    point_metrics = compute_time_point_classification_metrics(
        true_binary,
        pred_binary,
        timestamps,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
        well_coverage_threshold=DEFAULT_WELL_COVERAGE_THRESHOLD,
    )
    return {
        "id": entry.id,
        "video": entry.video,
        "start_time_s": entry.start_time_s,
        "duration_s": entry.duration_s,
        "gt_points_in_window": len(true_segments),
        "pred_points": len(pred_segments),
        "segment_metrics": segment_metrics,
        "point_metrics": point_metrics,
        "boundary_offsets": boundary_offsets(true_segments, pred_segments),
    }


def run_pipeline_to_csv(
    *,
    video_path: Path,
    entry: QualityEntry,
    csv_output_dir: Path,
    runner: RunnerConfig,
) -> Path:
    argv = _runner_argv(runner) + [
        "--video",
        str(video_path),
        "--start-time",
        str(int(entry.start_time_s)),
        "--duration",
        str(int(entry.duration_s)),
        "--write-csv",
        "--csv-output-dir",
        str(csv_output_dir),
        "--no-segment-video",
    ]
    argv.extend(runner.extra_args)
    proc = subprocess.run(argv, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{entry.id}: pipeline failed with exit code {proc.returncode}\n"
            f"command: {' '.join(argv)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    csv_path = csv_output_dir / f"{video_path.stem}_segments.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{entry.id}: pipeline succeeded but did not write expected CSV: {csv_path}"
        )
    return csv_path


def load_windowed_gt_segments(
    gt_path: Path,
    *,
    start_time_s: float,
    duration_s: float,
) -> list[tuple[float, float]]:
    end_time_s = start_time_s + duration_s
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    out: list[tuple[float, float]] = []
    for seg in data.get("segments", []):
        start = float(seg["start_time"])
        end = float(seg["end_time"])
        if end <= start:
            continue
        # Drop boundary-straddlers so the evaluation is about model behavior, not window cuts.
        if start < start_time_s < end or start < end_time_s < end:
            continue
        if start >= start_time_s and end <= end_time_s:
            out.append((start - start_time_s, end - start_time_s))
    return out


def load_segments_csv(path: Path) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                start = float(row["start_time"])
                end = float(row["end_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                segments.append((start, end))
    return segments


def intervals_to_binary(segments: Iterable[tuple[float, float]], timestamps: np.ndarray) -> np.ndarray:
    binary = np.zeros(timestamps.shape, dtype=int)
    frame_intervals = _frame_intervals_from_timestamps(timestamps)
    for idx, (frame_start, frame_end) in enumerate(frame_intervals):
        if any(_interval_overlap((frame_start, frame_end), seg) > 0.0 for seg in segments):
            binary[idx] = 1
    return binary


def boundary_offsets(
    true_segments: list[tuple[float, float]],
    pred_segments: list[tuple[float, float]],
) -> dict[str, Any]:
    matched: list[dict[str, float]] = []
    used_pred: set[int] = set()
    for true_seg in true_segments:
        best_idx = None
        best_iou = 0.0
        for pred_idx, pred_seg in enumerate(pred_segments):
            if pred_idx in used_pred:
                continue
            iou = _interval_iou(true_seg, pred_seg)
            if iou > best_iou:
                best_iou = iou
                best_idx = pred_idx
        if best_idx is None or best_iou <= 0.0:
            continue
        used_pred.add(best_idx)
        pred_seg = pred_segments[best_idx]
        matched.append(
            {
                "iou": best_iou,
                "start_offset_s": pred_seg[0] - true_seg[0],
                "end_offset_s": pred_seg[1] - true_seg[1],
            }
        )
    return {
        "matched_points": len(matched),
        "mean_start_offset_s": _mean([m["start_offset_s"] for m in matched]),
        "mean_end_offset_s": _mean([m["end_offset_s"] for m in matched]),
        "p90_abs_start_offset_s": _percentile_abs([m["start_offset_s"] for m in matched], 90),
        "p90_abs_end_offset_s": _percentile_abs([m["end_offset_s"] for m in matched], 90),
        "matches": matched,
    }


def _pool_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    count_keys = (
        "total_true_points",
        "total_pred_points",
        "well_classified_points",
        "cut_off_points",
        "missed_points",
        "false_detected_points",
        "unmatched_predicted_points",
    )
    pooled = {key: 0 for key in count_keys}
    for report in reports:
        metrics = report["point_metrics"]
        for key in count_keys:
            pooled[key] += int(metrics.get(key, 0))

    total_true = pooled["total_true_points"]
    total_pred = pooled["total_pred_points"]
    pooled.update(
        {
            "well_classified_rate": pooled["well_classified_points"] / total_true if total_true else 0.0,
            "cut_off_rate": pooled["cut_off_points"] / total_true if total_true else 0.0,
            "missed_rate": pooled["missed_points"] / total_true if total_true else 0.0,
            "false_detected_rate": pooled["false_detected_points"] / total_pred if total_pred else 0.0,
            "unmatched_predicted_rate": pooled["unmatched_predicted_points"] / total_pred if total_pred else 0.0,
            "segment_f1_mean": _mean([r["segment_metrics"]["segment_f1"] for r in reports]),
            "segment_recall_mean": _mean([r["segment_metrics"]["segment_recall"] for r in reports]),
            "segment_precision_mean": _mean([r["segment_metrics"]["segment_precision"] for r in reports]),
        }
    )
    return pooled


def _runner_argv(runner: RunnerConfig) -> list[str]:
    if runner.mode == "release":
        if runner.release_bin is None:
            raise ValueError("release runner requires release_bin")
        return [str(runner.release_bin), "--cli"]
    if runner.mode == "dev":
        return list(runner.command)
    raise ValueError(f"unknown runner mode: {runner.mode}")


def _runner_label(runner: RunnerConfig) -> dict[str, Any]:
    return {
        "mode": runner.mode,
        "release_bin": _path_label(runner.release_bin) if runner.release_bin is not None else None,
        "command": [_path_label(Path(arg)) if Path(arg).is_absolute() else arg for arg in runner.command],
        "extra_args": list(runner.extra_args),
    }


def _path_label(path: Path) -> str:
    return str(path) if not path.is_absolute() else path.name


def _timestamps(duration_s: float, fps: float) -> np.ndarray:
    step = 1.0 / fps
    count = max(1, int(round(duration_s * fps)))
    return np.arange(count, dtype=np.float64) * step


def _frame_intervals_from_timestamps(timestamps: np.ndarray) -> list[tuple[float, float]]:
    if timestamps.size == 1:
        return [(float(timestamps[0]), float(timestamps[0]) + 1.0)]
    step = max(1e-6, float(timestamps[1] - timestamps[0]))
    return [(float(t - step / 2.0), float(t + step / 2.0)) for t in timestamps]


def _interval_iou(seg_a: tuple[float, float], seg_b: tuple[float, float]) -> float:
    overlap = _interval_overlap(seg_a, seg_b)
    union = (seg_a[1] - seg_a[0]) + (seg_b[1] - seg_b[0]) - overlap
    return overlap / union if union > 0 else 0.0


def _interval_overlap(seg_a: tuple[float, float], seg_b: tuple[float, float]) -> float:
    return max(0.0, min(seg_a[1], seg_b[1]) - max(seg_a[0], seg_b[0]))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(mean(values)) if values else 0.0


def _percentile_abs(values: Iterable[float], percentile: float) -> float:
    values = [abs(v) for v in values]
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def runner_from_env(mode: str, extra_args: Iterable[str] = ()) -> RunnerConfig:
    if mode == "release":
        raw = os.environ.get("RALLYCLIP_RELEASE_BIN")
        if not raw:
            raise ValueError("set RALLYCLIP_RELEASE_BIN to use release mode")
        return RunnerConfig(mode="release", release_bin=Path(raw), extra_args=tuple(extra_args))
    return RunnerConfig(mode="dev", extra_args=tuple(extra_args))
