"""Evaluation for the boundary-heatmap head: stitch overlapping windows into
per-video timelines, decode segments from (pointness, startness, endness), and
score with the six-bin metric (docs/segment_eval_metrics.md).

The hybrid and peakpair decodes ARE the shipped runtime's (infer.heatmap_decode),
so offline numbers are the numbers the app produces. Training adds one mode the
runtime does not ship:
  - "pairdp": globally optimal start->end pairing by dynamic programming over
    boundary log-odds (see decode_pairdp). An experiment arm; not shipped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from infer.heatmap_decode import HeatmapDecodeConfig as RuntimeDecodeConfig
from infer.heatmap_decode import _merge_intervals, _pick_peaks, _soft_argmax_time
from infer.heatmap_decode import decode_heatmap_segments as runtime_decode
from training.metrics.frame import compute_frame_metrics
from training.metrics.segments6 import SixBinConfig, aggregate_six_bin, compute_six_bin

Interval = Tuple[float, float]

# train-config key -> decode field, for every knob a run config may set.
_TRAIN_KEYS = {
    "heatmap_decode_mode": "mode",
    "threshold": "threshold",
    "heatmap_peak_threshold": "peak_threshold",
    "heatmap_decode_window_frames": "refine_window_frames",
    "heatmap_nms_frames": "nms_frames",
    "heatmap_min_duration_sec": "min_duration_sec",
    "heatmap_max_duration_sec": "max_duration_sec",
    "heatmap_pointness_gate": "pointness_gate",
    "heatmap_smooth_sigma_frames": "smooth_sigma_frames",
    "heatmap_hybrid_min_duration_sec": "hybrid_min_duration_sec",
    "heatmap_merge_gap_sec": "merge_gap_sec",
    "heatmap_pair_penalty": "pair_penalty",
    "heatmap_pointness_weight": "pointness_weight",
}


@dataclass
class HeatmapDecodeConfig(RuntimeDecodeConfig):
    """The runtime decode config plus the training-only pairdp knobs."""

    # pairdp: per-segment cost subtracted from each segment's log-odds score. A
    # prior on how many segments exist -- without it the DP admits every pair that
    # scores positive and over-segments (878 preds for 460 GT at lambda=0).
    pair_penalty: float = 0.0
    # pairdp: weight on the pointness evidence for a candidate span. The span's
    # log-likelihood ratio of in-play vs out-of-play is sum_t logit(pointness[t]),
    # so this term lets the dense track say WHICH spans are real -- the one thing
    # two boundary heatmaps cannot express. 0.0 = pure boundary DP.
    pointness_weight: float = 0.0

    @classmethod
    def from_train_cfg(cls, train_cfg: Dict[str, Any], fps: float) -> "HeatmapDecodeConfig":
        """The one place a train config becomes a decode config (train loop,
        pipeline eval, and the run manifest all go through here)."""
        sigma_seconds = float(train_cfg.get("heatmap_sigma_seconds", 0.5))
        params: Dict[str, Any] = {"sigma_frames": sigma_seconds * float(fps)}
        for key, field_name in _TRAIN_KEYS.items():
            if train_cfg.get(key) is not None:
                params[field_name] = train_cfg[key]
        return cls(**params)

    def runtime_params(self) -> Dict[str, Any]:
        """Fields the shipped runtime understands, for the model manifest's
        postprocess.params."""
        return {f.name: getattr(self, f.name) for f in fields(RuntimeDecodeConfig)}


def gt_segments_from_targets(targets: np.ndarray, timestamps: np.ndarray) -> List[Interval]:
    """Runs of 1s -> (timestamp of first frame, timestamp of last frame)."""
    segments: List[Interval] = []
    n = len(targets)
    i = 0
    while i < n:
        if targets[i] < 0.5:
            i += 1
            continue
        j = i
        while j + 1 < n and targets[j + 1] >= 0.5:
            j += 1
        segments.append((float(timestamps[i]), float(timestamps[j])))
        i = j + 1
    return segments


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def decode_pairdp(
    pointness: np.ndarray,
    start_prob: np.ndarray,
    end_prob: np.ndarray,
    timestamps: np.ndarray,
    cfg: HeatmapDecodeConfig,
) -> List[Interval]:
    """Globally optimal start->end pairing by dynamic programming.

    peakpair walks starts left to right and takes the first unused end that
    clears min_duration. That commits irrevocably: one bad match shifts every
    later pairing. Here the segmentation is chosen jointly. A candidate segment
    scores the sum of its boundary LOG-ODDS:

        score(s, e) = logit(start_prob[s]) + logit(end_prob[e]) - pair_penalty

    which is self-calibrating -- a boundary with p > 0.5 contributes positively
    and one with p < 0.5 negatively, so the optimiser decides how many segments
    to emit. Maximising the total over all valid segmentations (non-overlapping,
    duration within [min, max]) is a longest-path problem on a DAG, solved
    exactly in O(m*n) over peak candidates.

    peak_threshold is only a candidate filter (keep it low); the log-odds
    objective does the real selection.
    """
    nms = cfg._nms()
    window = cfg._refine_window()
    s_idx = _pick_peaks(start_prob, cfg.peak_threshold, nms)
    e_idx = _pick_peaks(end_prob, cfg.peak_threshold, nms)
    if not s_idx or not e_idx:
        return []

    s_time = [_soft_argmax_time(start_prob, timestamps, i, window) for i in s_idx]
    e_time = [_soft_argmax_time(end_prob, timestamps, i, window) for i in e_idx]
    s_score = [_logit(start_prob[i]) for i in s_idx]
    e_score = [_logit(end_prob[i]) for i in e_idx]
    m, n = len(s_idx), len(e_idx)

    # Prefix sums of pointness log-odds, so a span's evidence is O(1) to query.
    cum = (np.concatenate(([0.0], np.cumsum([_logit(p) for p in pointness])))
           if cfg.pointness_weight else None)

    best_closed = [0.0] * (n + 1)      # best score using ends[:j], all segments closed
    back: List[Optional[Tuple[int, Optional[int]]]] = [None] * (n + 1)

    for j in range(1, n + 1):
        # option A: no segment closes at end j
        best_closed[j] = best_closed[j - 1]
        back[j] = (j - 1, None)
        # option B: a segment opened at some start i closes at end j
        et = e_time[j - 1]
        cand, cand_i, cand_pj = float("-inf"), None, None
        for i in range(m):
            dur = et - s_time[i]
            if dur < cfg.min_duration_sec:
                break                                    # s_time ascending -> no later i qualifies
            if dur > cfg.max_duration_sec:
                continue
            # the previous segment must close strictly before this start
            pj = _last_end_before(e_time, s_time[i], j - 1)
            val = best_closed[pj] + s_score[i] + e_score[j - 1] - cfg.pair_penalty
            if cum is not None:
                lo, hi = s_idx[i], e_idx[j - 1]
                val += cfg.pointness_weight * float(cum[hi + 1] - cum[lo])
            if val > cand:
                cand, cand_i, cand_pj = val, i, pj
        if cand_i is not None and cand > best_closed[j]:
            best_closed[j] = cand
            back[j] = (cand_pj, cand_i)

    segments: List[Interval] = []
    j = n
    while j > 0:
        prev_j, i = back[j]
        if i is not None:
            segments.append((s_time[i], e_time[j - 1]))
        j = prev_j
    segments.reverse()
    return _merge_intervals(segments, gap=cfg.merge_gap_sec)


def _last_end_before(e_time: List[float], t: float, upto: int) -> int:
    """Count of ends strictly before time t, capped at `upto` (index into best_closed)."""
    lo, hi = 0, upto
    while lo < hi:
        mid = (lo + hi) // 2
        if e_time[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def decode_heatmap_segments(
    pointness: np.ndarray,
    start_prob: np.ndarray,
    end_prob: np.ndarray,
    timestamps: np.ndarray,
    cfg: HeatmapDecodeConfig,
) -> List[Interval]:
    if cfg.mode == "pairdp":
        return decode_pairdp(pointness, start_prob, end_prob, timestamps, cfg)
    return runtime_decode(pointness, start_prob, end_prob, timestamps, cfg)


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

    # Frame metrics count overlapping windows once per window, not per frame.
    frame_metrics = compute_frame_metrics(
        targets.reshape(-1), point.reshape(-1).astype(np.float64), threshold=decode_cfg.threshold
    )

    stitched = _stitch_videos_heatmap(
        video_idx, frame_idx, timestamps, targets, point, start_p, end_p
    )
    per_video = []
    for tl in stitched.values():
        gt_segs = gt_segments_from_targets(tl["targets"], tl["timestamps"])
        pred_segs = decode_heatmap_segments(
            tl["pointness"], tl["start_prob"], tl["end_prob"], tl["timestamps"], decode_cfg
        )
        per_video.append(compute_six_bin(gt_segs, pred_segs, six_bin_cfg))

    metrics = {**frame_metrics, **aggregate_six_bin(per_video)}
    avg_loss = total_loss / max(batches, 1)
    return metrics, avg_loss
