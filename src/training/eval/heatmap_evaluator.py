"""Evaluation for the boundary-heatmap head: stitch overlapping windows into
per-video timelines, decode segments from (pointness, startness, endness), and
score with the six-bin metric (docs/segment_eval_metrics.md).

Two decode modes:
  - "hybrid" (default): detect points as runs of above-threshold pointness (the
    same robust detector the classic/e2e_seg heads use — one segment per detected
    point, no chance of losing a point to a missed boundary peak), then *refine*
    each run's start/end to sub-frame precision via soft-argmax of the
    start/end heatmap near the run edge. Targets the bad_segmentation failure
    (boundaries off 0.5–1.5s) without touching recall.
  - "peakpair": pure BSN-style — peak-pick startness and endness (with NMS),
    soft-argmax refine, greedy-pair start→next-end under a duration gate,
    optional pointness gate. More fragile (a point needs both peaks + a valid
    pair); kept selectable for comparison.
Both modes end with an overlap-merge of decoded segments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from training.eval.seg_evaluator import gt_segments_from_targets
from training.metrics.frame import compute_frame_metrics
from training.metrics.segments6 import SixBinConfig, aggregate_six_bin, compute_six_bin

Interval = Tuple[float, float]


@dataclass
class HeatmapDecodeConfig:
    mode: str = "hybrid"  # hybrid | peakpair
    threshold: float = 0.5  # pointness threshold (hybrid run detection; frame metrics)
    peak_threshold: float = 0.3  # start/end heatmap peak threshold (peakpair mode)
    sigma_frames: float = 2.5  # sets default refine / NMS windows
    refine_window_frames: Optional[int] = None  # default ceil(2*sigma)
    nms_frames: Optional[int] = None  # default ceil(sigma); min peak separation
    min_duration_sec: float = 0.3
    max_duration_sec: float = 60.0
    pointness_gate: Optional[float] = None  # peakpair mode; None disables
    # --- hybrid-mode false-positive filtering (both default to no-ops, so
    # existing hybrid results reproduce bit-for-bit) ---
    # Gaussian-smooth pointness before run detection. The classic decode has
    # always smoothed (sigma 1.5); hybrid never did, which penalises backbones
    # with noisier per-frame output (e.g. the TCN vs the naturally-smoothed LSTM).
    smooth_sigma_frames: Optional[float] = None
    # Drop decoded segments shorter than this. `min_duration_sec` above is only
    # consulted by peakpair's pairing loop; hybrid had no duration filter at all.
    hybrid_min_duration_sec: float = 0.0

    def _refine_window(self) -> int:
        return int(self.refine_window_frames if self.refine_window_frames is not None
                   else max(1, math.ceil(2.0 * self.sigma_frames)))

    def _nms(self) -> int:
        return int(self.nms_frames if self.nms_frames is not None
                   else max(1, math.ceil(self.sigma_frames)))


def _merge_intervals(segments: List[Interval]) -> List[Interval]:
    segments = sorted(s for s in segments if s[1] > s[0])
    merged: List[Interval] = []
    for seg in segments:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)
    return merged


def _soft_argmax_time(
    prob: np.ndarray, timestamps: np.ndarray, center: int, window: int
) -> float:
    """Probability-weighted mean of frame times in [center-window, center+window].
    Falls back to the plain window-centre time if the local heatmap mass vanishes
    (so a frame with no boundary signal still yields the run-edge time)."""
    lo = max(0, center - window)
    hi = min(len(prob), center + window + 1)
    w = prob[lo:hi]
    ts = timestamps[lo:hi]
    total = float(w.sum())
    if total <= 1e-9:
        return float(ts.mean())
    return float(np.average(ts, weights=w))


def _pick_peaks(prob: np.ndarray, threshold: float, nms_frames: int) -> List[int]:
    """Local maxima (>= both neighbours) above threshold, then NMS: keep peaks in
    descending prob, drop any within nms_frames of an already-kept, stronger peak."""
    n = len(prob)
    cand: List[int] = []
    for i in range(n):
        if prob[i] < threshold:
            continue
        left_ok = i == 0 or prob[i] >= prob[i - 1]
        right_ok = i == n - 1 or prob[i] >= prob[i + 1]
        if left_ok and right_ok:
            cand.append(i)
    cand.sort(key=lambda i: float(prob[i]), reverse=True)
    kept: List[int] = []
    for i in cand:
        if all(abs(i - k) > nms_frames for k in kept):
            kept.append(i)
    kept.sort()
    return kept


def _runs_above(prob: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """(first_idx, last_idx) of each contiguous run of prob >= threshold."""
    runs: List[Tuple[int, int]] = []
    n = len(prob)
    i = 0
    while i < n:
        if prob[i] < threshold:
            i += 1
            continue
        j = i
        while j + 1 < n and prob[j + 1] >= threshold:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _gaussian_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian filter, numpy-only. Kept identical to infer.inference's
    gaussian_filter1d so training and the shipped runtime stay in parity."""
    if sigma <= 0:
        return data.copy()
    radius = int(3 * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(data, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(data.dtype)


def decode_hybrid(
    pointness: np.ndarray,
    start_prob: np.ndarray,
    end_prob: np.ndarray,
    timestamps: np.ndarray,
    cfg: HeatmapDecodeConfig,
) -> List[Interval]:
    window = cfg._refine_window()
    # Run detection sees the (optionally) smoothed track; the start/end heatmaps
    # are left raw so boundary refinement keeps its sub-frame sharpness.
    detect_track = (
        pointness if cfg.smooth_sigma_frames is None
        else _gaussian_smooth(pointness, float(cfg.smooth_sigma_frames))
    )
    segments: List[Interval] = []
    for i, j in _runs_above(detect_track, cfg.threshold):
        s = _soft_argmax_time(start_prob, timestamps, i, window)
        e = _soft_argmax_time(end_prob, timestamps, j, window)
        # Refinement must not invert or escape the detected run's rough span.
        if e <= s:
            s, e = float(timestamps[i]), float(timestamps[j])
        if e > s:
            segments.append((s, e))
    merged = _merge_intervals(segments)
    # Duration floor applied AFTER merging, so adjacent fragments that together
    # form a real point are not discarded individually.
    if cfg.hybrid_min_duration_sec > 0:
        merged = [(s, e) for s, e in merged if (e - s) >= cfg.hybrid_min_duration_sec]
    return merged


def decode_peakpair(
    pointness: np.ndarray,
    start_prob: np.ndarray,
    end_prob: np.ndarray,
    timestamps: np.ndarray,
    cfg: HeatmapDecodeConfig,
) -> List[Interval]:
    window = cfg._refine_window()
    nms = cfg._nms()
    start_peaks = _pick_peaks(start_prob, cfg.peak_threshold, nms)
    end_peaks = _pick_peaks(end_prob, cfg.peak_threshold, nms)
    start_times = sorted(_soft_argmax_time(start_prob, timestamps, p, window) for p in start_peaks)
    end_times = sorted(_soft_argmax_time(end_prob, timestamps, p, window) for p in end_peaks)

    segments: List[Interval] = []
    ei = 0
    used = [False] * len(end_times)
    for st in start_times:
        k = ei
        while k < len(end_times) and (used[k] or end_times[k] < st + cfg.min_duration_sec):
            k += 1
        if k >= len(end_times):
            continue
        et = end_times[k]
        if et - st > cfg.max_duration_sec:
            continue
        used[k] = True
        ei = k + 1
        if cfg.pointness_gate is not None:
            lo = np.searchsorted(timestamps, st)
            hi = np.searchsorted(timestamps, et)
            span = pointness[lo:hi + 1]
            if span.size and float(span.mean()) < cfg.pointness_gate:
                continue
        segments.append((st, et))
    return _merge_intervals(segments)


def decode_heatmap_segments(
    pointness: np.ndarray,
    start_prob: np.ndarray,
    end_prob: np.ndarray,
    timestamps: np.ndarray,
    cfg: HeatmapDecodeConfig,
) -> List[Interval]:
    if cfg.mode == "hybrid":
        return decode_hybrid(pointness, start_prob, end_prob, timestamps, cfg)
    if cfg.mode == "peakpair":
        return decode_peakpair(pointness, start_prob, end_prob, timestamps, cfg)
    raise ValueError(f"Unknown decode mode: {cfg.mode!r} (expected hybrid | peakpair)")


def _stitch_videos_heatmap(
    video_idx: np.ndarray,
    frame_idx: np.ndarray,
    timestamps: np.ndarray,
    targets: np.ndarray,
    pointness: np.ndarray,
    start_prob: np.ndarray,
    end_prob: np.ndarray,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Plain per-frame mean of each probability track across overlapping windows,
    keyed by (video, native frame index); returns per-video sorted timelines."""
    # acc = [ts, target, sum_point, sum_start, sum_end, count]
    videos: Dict[int, Dict[int, List]] = {}
    n_seq, seq_len = targets.shape
    for s in range(n_seq):
        vid = int(video_idx[s])
        frames = videos.setdefault(vid, {})
        for t in range(seq_len):
            key = int(frame_idx[s, t])
            acc = frames.get(key)
            if acc is None:
                frames[key] = [
                    float(timestamps[s, t]), float(targets[s, t]),
                    float(pointness[s, t]), float(start_prob[s, t]), float(end_prob[s, t]), 1,
                ]
            else:
                acc[2] += float(pointness[s, t])
                acc[3] += float(start_prob[s, t])
                acc[4] += float(end_prob[s, t])
                acc[5] += 1

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for vid, frames in videos.items():
        keys = sorted(frames)
        counts = np.array([frames[k][5] for k in keys], dtype=np.float64)
        out[vid] = {
            "timestamps": np.array([frames[k][0] for k in keys], dtype=np.float64),
            "targets": np.array([frames[k][1] for k in keys], dtype=np.float32),
            "pointness": np.array([frames[k][2] for k in keys], dtype=np.float64) / counts,
            "start_prob": np.array([frames[k][3] for k in keys], dtype=np.float64) / counts,
            "end_prob": np.array([frames[k][4] for k in keys], dtype=np.float64) / counts,
        }
    return out


def evaluate_heatmap_model(
    model: torch.nn.Module,
    h5_path: Path,
    device: torch.device,
    criterion: torch.nn.Module,
    decode_cfg: HeatmapDecodeConfig | None = None,
    six_bin_cfg: SixBinConfig | None = None,
    batch_size: int = 32,
) -> Tuple[Dict[str, float], float]:
    decode_cfg = decode_cfg or HeatmapDecodeConfig()
    model.eval()

    with h5py.File(h5_path, "r") as h5f:
        features = h5f["features"][:]
        targets = h5f["targets"][:].astype(np.float32)
        video_idx = h5f["sequence_video_index"][:]
        frame_idx = h5f["sequence_frame_index"][:]
        timestamps = h5f["sequence_timestamps"][:]

    n_seq = features.shape[0]
    point = np.zeros_like(targets, dtype=np.float32)
    start_p = np.zeros_like(targets, dtype=np.float32)
    end_p = np.zeros_like(targets, dtype=np.float32)
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            feats = torch.from_numpy(features[start:end]).float().to(device)
            targs = torch.from_numpy(targets[start:end]).to(device)
            p_logits, s_logits, e_logits = model(feats)
            loss, _ = criterion(p_logits, s_logits, e_logits, targs)
            total_loss += float(loss.item())
            batches += 1
            point[start:end] = torch.sigmoid(p_logits).cpu().numpy()
            start_p[start:end] = torch.sigmoid(s_logits).cpu().numpy()
            end_p[start:end] = torch.sigmoid(e_logits).cpu().numpy()

    frame_metrics = compute_frame_metrics(
        targets.reshape(-1), point.reshape(-1).astype(np.float64), threshold=decode_cfg.threshold
    )

    stitched = _stitch_videos_heatmap(
        video_idx, frame_idx, timestamps, targets, point, start_p, end_p
    )
    per_video = []
    for vid, tl in stitched.items():
        gt_segs = gt_segments_from_targets(tl["targets"], tl["timestamps"])
        pred_segs = decode_heatmap_segments(
            tl["pointness"], tl["start_prob"], tl["end_prob"], tl["timestamps"], decode_cfg
        )
        per_video.append(compute_six_bin(gt_segs, pred_segs, six_bin_cfg))

    metrics = {**frame_metrics, **aggregate_six_bin(per_video)}
    avg_loss = total_loss / max(batches, 1)
    return metrics, avg_loss
