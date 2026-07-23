"""Runtime heatmap decode + multi-track windowed inference.

Locks the torch-free runtime decode (infer.heatmap_decode) and the N-track
windowed-average runner (infer.inference) that feed the boundary-heatmap
production pipeline. The decode is a verbatim port of the training-side
evaluator, so these mirror the shapes the offline six-bin numbers were measured
on.
"""
from __future__ import annotations

import numpy as np
import pytest

from infer.heatmap_decode import (
    HeatmapDecodeConfig,
    _merge_intervals,
    _runs_above,
    _soft_argmax_time,
    decode_heatmap_segments,
    decode_hybrid,
    decode_peakpair,
)
from infer.inference import (
    _order_heatmap_outputs,
    run_multitrack_windowed_inference_stream,
    run_windowed_inference_average_stream,
)

FPS = 5.0


def _timestamps(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / FPS


# ---------------------------------------------------------------- decode helpers


def test_runs_above_finds_contiguous_runs():
    prob = np.array([0.1, 0.9, 0.9, 0.1, 0.8, 0.1], dtype=np.float64)
    assert _runs_above(prob, 0.5) == [(1, 2), (4, 4)]


def test_runs_above_empty_when_all_below():
    assert _runs_above(np.zeros(10), 0.5) == []


def test_soft_argmax_single_spike_is_exact():
    ts = _timestamps(20)
    prob = np.zeros(20)
    prob[5] = 0.9  # lone spike -> weighted mean is exactly that frame's time
    assert _soft_argmax_time(prob, ts, center=5, window=5) == pytest.approx(ts[5])


def test_soft_argmax_falls_back_to_center_time_when_no_mass():
    ts = _timestamps(20)
    prob = np.zeros(20)  # no heatmap mass anywhere
    # mean of the window's timestamps [3..7] -> ts[5]
    assert _soft_argmax_time(prob, ts, center=5, window=2) == pytest.approx(ts[3:8].mean())


def test_soft_argmax_interpolates_between_frames():
    ts = _timestamps(20)
    prob = np.zeros(20)
    prob[5] = 1.0
    prob[6] = 1.0  # equal mass on two adjacent frames -> midpoint time
    got = _soft_argmax_time(prob, ts, center=5, window=3)
    assert got == pytest.approx((ts[5] + ts[6]) / 2)


def test_merge_intervals_merges_touching_and_overlapping():
    assert _merge_intervals([(0.0, 1.0), (1.0, 2.0), (5.0, 6.0)]) == [(0.0, 2.0), (5.0, 6.0)]
    assert _merge_intervals([(0.0, 3.0), (1.0, 2.0)]) == [(0.0, 3.0)]
    assert _merge_intervals([(2.0, 1.0)]) == []  # degenerate dropped


# ---------------------------------------------------------------- hybrid decode


def test_decode_hybrid_single_point_recovers_boundaries():
    n = 20
    ts = _timestamps(n)
    pointness = np.full(n, 0.1)
    pointness[5:11] = 0.9  # run frames 5..10
    start_prob = np.zeros(n)
    start_prob[5] = 0.9
    end_prob = np.zeros(n)
    end_prob[10] = 0.9
    cfg = HeatmapDecodeConfig(mode="hybrid", threshold=0.5)
    segs = decode_hybrid(pointness, start_prob, end_prob, ts, cfg)
    assert len(segs) == 1
    assert segs[0] == pytest.approx((ts[5], ts[10]))


def test_decode_hybrid_two_points_no_cross_pairing():
    n = 24
    ts = _timestamps(n)
    pointness = np.full(n, 0.1)
    pointness[3:6] = 0.9   # point A: frames 3..5
    pointness[14:18] = 0.9  # point B: frames 14..17
    start_prob = np.zeros(n)
    start_prob[3] = 0.9
    start_prob[14] = 0.9
    end_prob = np.zeros(n)
    end_prob[5] = 0.9
    end_prob[17] = 0.9
    cfg = HeatmapDecodeConfig(mode="hybrid", threshold=0.5)
    segs = decode_hybrid(pointness, start_prob, end_prob, ts, cfg)
    assert len(segs) == 2
    assert segs[0] == pytest.approx((ts[3], ts[5]))
    assert segs[1] == pytest.approx((ts[14], ts[17]))


def test_decode_hybrid_falls_back_to_run_edges_when_refine_inverts():
    # start heatmap mass sits AFTER the end heatmap mass -> refined s>=e; decode
    # must fall back to the raw run-edge timestamps rather than emit nothing/garbage.
    n = 20
    ts = _timestamps(n)
    pointness = np.full(n, 0.1)
    pointness[5:11] = 0.9
    start_prob = np.zeros(n)
    start_prob[10] = 0.9  # inverted: "start" peak at the run end
    end_prob = np.zeros(n)
    end_prob[5] = 0.9      # inverted: "end" peak at the run start
    cfg = HeatmapDecodeConfig(mode="hybrid", threshold=0.5)
    segs = decode_hybrid(pointness, start_prob, end_prob, ts, cfg)
    assert len(segs) == 1
    assert segs[0] == pytest.approx((ts[5], ts[10]))  # raw run edges


def test_decode_hybrid_no_points_returns_empty():
    n = 20
    ts = _timestamps(n)
    cfg = HeatmapDecodeConfig(mode="hybrid", threshold=0.5)
    assert decode_hybrid(np.full(n, 0.1), np.zeros(n), np.zeros(n), ts, cfg) == []


# -------------------------------------------------------------- peakpair decode


def test_decode_peakpair_pairs_start_to_next_end():
    n = 24
    ts = _timestamps(n)
    pointness = np.full(n, 0.9)
    start_prob = np.zeros(n)
    start_prob[4] = 0.9
    end_prob = np.zeros(n)
    end_prob[12] = 0.9
    cfg = HeatmapDecodeConfig(mode="peakpair", peak_threshold=0.3, min_duration_sec=0.3)
    segs = decode_peakpair(pointness, start_prob, end_prob, ts, cfg)
    assert len(segs) == 1
    assert segs[0] == pytest.approx((ts[4], ts[12]))


def test_decode_peakpair_rejects_pairs_over_max_duration():
    n = 40
    ts = _timestamps(n)
    pointness = np.full(n, 0.9)
    start_prob = np.zeros(n)
    start_prob[2] = 0.9
    end_prob = np.zeros(n)
    end_prob[38] = 0.9
    # 36 frames / 5 fps = 7.2s > max 1.0s -> rejected
    cfg = HeatmapDecodeConfig(mode="peakpair", peak_threshold=0.3, max_duration_sec=1.0)
    assert decode_peakpair(pointness, start_prob, end_prob, ts, cfg) == []


def test_decode_dispatch_unknown_mode_raises():
    n = 10
    ts = _timestamps(n)
    cfg = HeatmapDecodeConfig(mode="nonsense")
    with pytest.raises(ValueError, match="Unknown decode mode"):
        decode_heatmap_segments(np.zeros(n), np.zeros(n), np.zeros(n), ts, cfg)


# ------------------------------------------------- ONNX output ordering mapping


def test_order_heatmap_outputs_by_name():
    # scrambled export order -> remapped to [point, start, end]
    names = ["end_heatmap_logit", "pointness_logit", "start_heatmap_logit"]
    assert _order_heatmap_outputs(names) == [1, 2, 0]


def test_order_heatmap_outputs_positional_fallback():
    names = ["out0", "out1", "out2"]  # uninformative -> declared order
    assert _order_heatmap_outputs(names) == [0, 1, 2]


# ------------------------------------------------ multi-track windowed inference


def test_multitrack_matches_single_track_for_one_track():
    # The N-track runner must reduce to the single-track path exactly (same
    # windowing + float32 accumulation) when K=1.
    rng = np.random.default_rng(0)
    rows = [rng.standard_normal(8).astype(np.float32) for _ in range(37)]
    L, overlap = 10, 5

    def win_1(window):  # [L, F] -> [L]
        return window.sum(axis=1)

    def win_k(window):  # [L, F] -> [1, L]
        return window.sum(axis=1)[None, :]

    single = run_windowed_inference_average_stream(list(rows), win_1, L, overlap)
    multi = run_multitrack_windowed_inference_stream(list(rows), win_k, L, overlap, num_tracks=1)
    assert multi.shape == (1, len(rows))
    np.testing.assert_allclose(multi[0], single, rtol=0, atol=1e-6)


def test_multitrack_averages_overlapping_windows():
    # Constant per-track output per window; overlapped frames must average to the
    # same constant, and the shape is (num_tracks, num_frames).
    n, L, overlap = 25, 10, 5
    rows = [np.ones(4, dtype=np.float32) for _ in range(n)]

    def win_k(window):
        # track t emits the constant (t+1) across the whole window
        return np.stack([np.full(L, t + 1.0, dtype=np.float32) for t in range(3)], axis=0)

    out = run_multitrack_windowed_inference_stream(list(rows), win_k, L, overlap, num_tracks=3)
    assert out.shape == (3, n)
    np.testing.assert_allclose(out[0], 1.0)
    np.testing.assert_allclose(out[1], 2.0)
    np.testing.assert_allclose(out[2], 3.0)


def test_multitrack_rejects_bad_window_shape():
    rows = [np.ones(4, dtype=np.float32) for _ in range(15)]

    def bad_win(window):
        return np.zeros((2, window.shape[0]))  # says 2 tracks, runner expects 3

    with pytest.raises(ValueError, match="window output shape mismatch"):
        run_multitrack_windowed_inference_stream(list(rows), bad_win, 10, 5, num_tracks=3)


def test_multitrack_raises_when_video_too_short():
    rows = [np.ones(4, dtype=np.float32) for _ in range(5)]  # < seq_len

    def win_k(window):
        return np.zeros((3, window.shape[0]), dtype=np.float32)

    with pytest.raises(ValueError, match="too short"):
        run_multitrack_windowed_inference_stream(list(rows), win_k, 10, 5, num_tracks=3)
