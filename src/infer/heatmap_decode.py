"""Torch-free decode for the boundary-heatmap head.

The runtime counterpart of ``training/eval/heatmap_evaluator``'s decode: given
per-frame ``(pointness, start_prob, end_prob)`` probability tracks already
stitched onto a single video timeline plus each frame's timestamp (seconds),
turn them into point intervals ``(start_s, end_s)`` in **floating-point
seconds** (sub-frame precision preserved via soft-argmax).

Kept byte-for-byte in step with the training decode so offline six-bin numbers
transfer to production; the only thing dropped here is the torch/h5py model-eval
scaffolding, which the runtime does not need. No torch import — this module runs
in the shipped ONNX-only runtime.

Two decode modes:
  - "hybrid" (default): detect points as runs of above-threshold pointness (the
    same robust detector the classic head uses -- one segment per detected
    point, no chance of losing a point to a missed boundary peak), then *refine*
    each run's start/end to sub-frame precision via soft-argmax of the
    start/end heatmap near the run edge. Targets boundary error without touching
    recall.
  - "peakpair": pure BSN-style -- peak-pick startness and endness (with NMS),
    soft-argmax refine, greedy-pair start->next-end under a duration gate,
    optional pointness gate. More fragile; kept selectable for parity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

Interval = Tuple[float, float]


@dataclass
class HeatmapDecodeConfig:
    mode: str = "hybrid"  # hybrid | peakpair
    threshold: float = 0.5  # pointness threshold (hybrid run detection)
    peak_threshold: float = 0.3  # start/end heatmap peak threshold (peakpair mode)
    sigma_frames: float = 2.5  # sets default refine / NMS windows
    refine_window_frames: Optional[int] = None  # default ceil(2*sigma)
    nms_frames: Optional[int] = None  # default ceil(sigma); min peak separation
    min_duration_sec: float = 0.3
    max_duration_sec: float = 60.0
    pointness_gate: Optional[float] = None  # peakpair mode; None disables

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


def decode_hybrid(
    pointness: np.ndarray,
    start_prob: np.ndarray,
    end_prob: np.ndarray,
    timestamps: np.ndarray,
    cfg: HeatmapDecodeConfig,
) -> List[Interval]:
    window = cfg._refine_window()
    segments: List[Interval] = []
    for i, j in _runs_above(pointness, cfg.threshold):
        s = _soft_argmax_time(start_prob, timestamps, i, window)
        e = _soft_argmax_time(end_prob, timestamps, j, window)
        # Refinement must not invert or escape the detected run's rough span.
        if e <= s:
            s, e = float(timestamps[i]), float(timestamps[j])
        if e <= s:
            continue
        dur = e - s
        if dur < cfg.min_duration_sec or dur > cfg.max_duration_sec:
            continue
        segments.append((s, e))
    return _merge_intervals(segments)


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
        if cfg.pointness_gate is not None:
            lo = np.searchsorted(timestamps, st)
            hi = np.searchsorted(timestamps, et)
            span = pointness[lo:hi + 1]
            if span.size and float(span.mean()) < cfg.pointness_gate:
                # Do not consume the end event: a later start may pair with it.
                continue
        used[k] = True
        ei = k + 1
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
