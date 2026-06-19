"""Evaluation for the per-frame segment head: stitch overlapping windows back into
per-video timelines, decode segments from (pointness, d_start, d_end), and score
with the six-bin metric (docs/segment_eval_metrics.md)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

from training.metrics.frame import compute_frame_metrics
from training.metrics.segments6 import SixBinConfig, aggregate_six_bin, compute_six_bin

Interval = Tuple[float, float]


@dataclass
class DecodeConfig:
    threshold: float = 0.5
    vote: str = "mean"  # "mean" | "median" (probability-weighted)
    # Cross-window offset aggregation (Step 1). "uniform" == plain mean (parity with
    # the original stitch). Others weight each window's d_start/d_end contribution for
    # a frame; the gate probability stays a plain mean regardless.
    #   prob          -> weight by that window's pointness prob for the frame
    #   inwindow      -> downweight (by extrapolation_gamma) windows where the predicted
    #                    boundary t +/- offset falls outside the window's own time span
    #   prob_inwindow -> product of the two
    offset_weight: str = "uniform"  # uniform | prob | inwindow | prob_inwindow
    extrapolation_gamma: float = 0.1  # weight for out-of-window (extrapolated) estimates


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    idx = int(np.searchsorted(cum, cum[-1] / 2.0))
    return float(values[order][min(idx, len(values) - 1)])


def decode_segments(
    probs: np.ndarray,
    d_start: np.ndarray,
    d_end: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 0.5,
    vote: str = "mean",
) -> List[Interval]:
    """Group consecutive above-threshold frames; each contributes a proposed
    (t - d_start, t + d_end); the group's segment is the probability-weighted
    mean (or median) of the votes. Overlapping decoded segments are merged."""
    above = probs >= threshold
    segments: List[Interval] = []
    n = len(probs)
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        w = probs[i : j + 1]
        starts = timestamps[i : j + 1] - d_start[i : j + 1]
        ends = timestamps[i : j + 1] + d_end[i : j + 1]
        if vote == "median":
            s = _weighted_median(starts, w)
            e = _weighted_median(ends, w)
        else:
            s = float(np.average(starts, weights=w))
            e = float(np.average(ends, weights=w))
        if e > s:
            segments.append((s, e))
        i = j + 1

    segments.sort()
    merged: List[Interval] = []
    for seg in segments:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)
    return merged


def gt_segments_from_targets(targets: np.ndarray, timestamps: np.ndarray) -> List[Interval]:
    """Runs of 1s -> (timestamp of first frame, timestamp of last frame).
    Consistent with training, where offsets are measured to the first/last
    in-point frame."""
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


def _offset_weight(
    prob: float,
    pred_start: float,
    pred_end: float,
    win_lo: float,
    win_hi: float,
    mode: str,
    gamma: float,
) -> float:
    """Weight for one window's offset estimate of one frame. The gate prob is always
    a plain mean; only the d_start/d_end tracks use this."""
    if mode == "uniform":
        return 1.0
    in_window = (pred_start >= win_lo - 1e-9) and (pred_end <= win_hi + 1e-9)
    if mode == "prob":
        return float(prob)
    if mode == "inwindow":
        return 1.0 if in_window else gamma
    if mode == "prob_inwindow":
        return float(prob) * (1.0 if in_window else gamma)
    raise ValueError(f"Unknown offset_weight mode: {mode}")


def _stitch_videos(
    video_idx: np.ndarray,
    frame_idx: np.ndarray,
    timestamps: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray,
    d_start: np.ndarray,
    d_end: np.ndarray,
    offset_weight: str = "uniform",
    extrapolation_gamma: float = 0.1,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Aggregate per-frame outputs across overlapping windows, keyed by
    (video, native frame index), and return per-video sorted timelines. The gate
    probability is a plain mean; d_start/d_end are a weighted mean per offset_weight.
    With offset_weight="uniform" this is identical to a plain mean of all tracks."""
    # acc = [ts, target, sum_prob, sum_w*d_start, sum_w*d_end, count, sum_w]
    videos: Dict[int, Dict[int, List]] = {}
    n_seq, seq_len = targets.shape
    for s in range(n_seq):
        vid = int(video_idx[s])
        frames = videos.setdefault(vid, {})
        win_lo = float(timestamps[s, 0])
        win_hi = float(timestamps[s, seq_len - 1])
        for t in range(seq_len):
            key = int(frame_idx[s, t])
            p = float(probs[s, t])
            ds = float(d_start[s, t])
            de = float(d_end[s, t])
            w = _offset_weight(
                p, timestamps[s, t] - ds, timestamps[s, t] + de, win_lo, win_hi, offset_weight, extrapolation_gamma
            )
            acc = frames.get(key)
            if acc is None:
                frames[key] = [timestamps[s, t], targets[s, t], p, w * ds, w * de, 1, w]
            else:
                acc[2] += p
                acc[3] += w * ds
                acc[4] += w * de
                acc[5] += 1
                acc[6] += w

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for vid, frames in videos.items():
        keys = sorted(frames)
        ts = np.array([frames[k][0] for k in keys], dtype=np.float64)
        tg = np.array([frames[k][1] for k in keys], dtype=np.float32)
        counts = np.array([frames[k][5] for k in keys], dtype=np.float64)
        sum_w = np.array([frames[k][6] for k in keys], dtype=np.float64)
        # Fall back to a plain count-mean for any frame whose total offset weight
        # vanished (e.g. prob-weighting where every window saw it as off-point).
        denom = np.where(sum_w > 1e-9, sum_w, counts)
        pr = np.array([frames[k][2] for k in keys], dtype=np.float64) / counts
        ds = np.array([frames[k][3] for k in keys], dtype=np.float64) / denom
        de = np.array([frames[k][4] for k in keys], dtype=np.float64) / denom
        out[vid] = {"timestamps": ts, "targets": tg, "probs": pr, "d_start": ds, "d_end": de}
    return out


def evaluate_seg_model(
    model: torch.nn.Module,
    h5_path: Path,
    device: torch.device,
    criterion: torch.nn.Module,
    decode_cfg: DecodeConfig | None = None,
    six_bin_cfg: SixBinConfig | None = None,
    batch_size: int = 32,
) -> Tuple[Dict[str, float], float]:
    decode_cfg = decode_cfg or DecodeConfig()
    model.eval()

    with h5py.File(h5_path, "r") as h5f:
        features = h5f["features"][:]
        targets = h5f["targets"][:].astype(np.float32)
        video_idx = h5f["sequence_video_index"][:]
        frame_idx = h5f["sequence_frame_index"][:]
        timestamps = h5f["sequence_timestamps"][:]

    n_seq = features.shape[0]
    probs = np.zeros_like(targets, dtype=np.float32)
    d_start = np.zeros_like(targets, dtype=np.float32)
    d_end = np.zeros_like(targets, dtype=np.float32)
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            feats = torch.from_numpy(features[start:end]).float().to(device)
            targs = torch.from_numpy(targets[start:end]).to(device)
            logits, ds, de = model(feats)
            loss, _ = criterion(logits, ds, de, targs)
            total_loss += float(loss.item())
            batches += 1
            probs[start:end] = torch.sigmoid(logits).cpu().numpy()
            d_start[start:end] = ds.cpu().numpy()
            d_end[start:end] = de.cpu().numpy()

    frame_metrics = compute_frame_metrics(
        targets.reshape(-1), probs.reshape(-1).astype(np.float64), threshold=decode_cfg.threshold
    )

    stitched = _stitch_videos(
        video_idx, frame_idx, timestamps, targets, probs, d_start, d_end,
        offset_weight=decode_cfg.offset_weight,
        extrapolation_gamma=decode_cfg.extrapolation_gamma,
    )
    per_video = []
    for vid, tl in stitched.items():
        gt_segs = gt_segments_from_targets(tl["targets"], tl["timestamps"])
        pred_segs = decode_segments(
            tl["probs"],
            tl["d_start"],
            tl["d_end"],
            tl["timestamps"],
            threshold=decode_cfg.threshold,
            vote=decode_cfg.vote,
        )
        per_video.append(compute_six_bin(gt_segs, pred_segs, six_bin_cfg))

    metrics = {**frame_metrics, **aggregate_six_bin(per_video)}
    avg_loss = total_loss / max(batches, 1)
    return metrics, avg_loss
