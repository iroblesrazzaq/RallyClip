"""Dynamic-programming start/end pairing (decode mode `pairdp`)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.eval.heatmap_evaluator import (  # noqa: E402
    HeatmapDecodeConfig, decode_pairdp, decode_peakpair,
)

N = 60


def _spikes(peaks: dict[int, float]) -> np.ndarray:
    """Flat 0.02 background with isolated spikes at the given frames."""
    a = np.full(N, 0.02, dtype=np.float64)
    for i, p in peaks.items():
        a[i] = p
    return a


def _cfg(**kw):
    base = dict(mode="pairdp", peak_threshold=0.05, sigma_frames=1.0,
                refine_window_frames=0, nms_frames=1,
                min_duration_sec=2.0, max_duration_sec=60.0, pointness_gate=None)
    base.update(kw)
    return HeatmapDecodeConfig(**base)


TS = np.arange(N, dtype=np.float64)   # 1 second per frame, keeps the arithmetic readable


def test_recovers_two_clean_points():
    s = _spikes({10: 0.99, 30: 0.99})
    e = _spikes({20: 0.99, 40: 0.99})
    got = decode_pairdp(np.zeros(N), s, e, TS, _cfg())
    assert got == [(10.0, 20.0), (30.0, 40.0)]


def test_beats_greedy_when_a_spurious_start_would_steal_an_end():
    """The cascade that sinks greedy: a weak spurious start at t=0 grabs the first
    end, shifting every later pairing. DP scores the whole segmentation, so it
    simply leaves the weak start unpaired."""
    s = _spikes({0: 0.55, 10: 0.99, 30: 0.99})
    e = _spikes({20: 0.99, 40: 0.99})
    truth = [(10.0, 20.0), (30.0, 40.0)]

    dp = decode_pairdp(np.zeros(N), s, e, TS, _cfg())
    greedy = decode_peakpair(np.zeros(N), s, e, TS, _cfg(mode="peakpair"))

    assert dp == truth
    assert greedy != truth, "if greedy also solves this, the test no longer discriminates"


def test_weak_boundaries_are_dropped_by_the_log_odds_objective():
    """p < 0.5 on both ends makes a segment's score negative, so it is excluded
    without any threshold tuning."""
    s = _spikes({10: 0.99, 30: 0.20})
    e = _spikes({20: 0.99, 40: 0.20})
    got = decode_pairdp(np.zeros(N), s, e, TS, _cfg())
    assert got == [(10.0, 20.0)]


def test_duration_bounds_are_respected():
    s = _spikes({10: 0.99})
    e = _spikes({11: 0.99})            # 1s apart, below min_duration 2.0
    assert decode_pairdp(np.zeros(N), s, e, TS, _cfg()) == []
    assert decode_pairdp(np.zeros(N), s, e, TS, _cfg(min_duration_sec=0.5)) == [(10.0, 11.0)]
    s2 = _spikes({5: 0.99})
    e2 = _spikes({50: 0.99})
    assert decode_pairdp(np.zeros(N), s2, e2, TS, _cfg(max_duration_sec=10.0)) == []


def test_segments_never_overlap():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = np.clip(rng.random(N), 0.01, 0.99)
        e = np.clip(rng.random(N), 0.01, 0.99)
        got = decode_pairdp(np.zeros(N), s, e, TS, _cfg())
        for a, b in zip(got, got[1:]):
            assert a[1] <= b[0], f"overlap: {a} then {b}"
        for st, en in got:
            assert st < en


def test_no_candidates_yields_no_segments():
    flat = np.full(N, 0.001)
    assert decode_pairdp(np.zeros(N), flat, flat, TS, _cfg()) == []


def test_dp_total_score_is_at_least_greedy_score():
    """DP is exact, so on any input its objective must be >= greedy's."""
    def score(segs, s, e):
        tot = 0.0
        for st, en in segs:
            tot += np.log(s[int(st)] / (1 - s[int(st)])) + np.log(e[int(en)] / (1 - e[int(en)]))
        return tot

    rng = np.random.default_rng(7)
    for _ in range(20):
        s = _spikes({int(i): float(p) for i, p in
                     zip(rng.choice(N, 5, replace=False), rng.uniform(0.3, 0.99, 5))})
        e = _spikes({int(i): float(p) for i, p in
                     zip(rng.choice(N, 5, replace=False), rng.uniform(0.3, 0.99, 5))})
        dp = decode_pairdp(np.zeros(N), s, e, TS, _cfg())
        gr = decode_peakpair(np.zeros(N), s, e, TS, _cfg(mode="peakpair"))
        assert score(dp, s, e) >= score(gr, s, e) - 1e-9


def test_pair_penalty_monotonically_reduces_segment_count():
    """lambda is a prior on how many segments exist. Raising it must never add
    segments, and a large enough lambda must suppress all of them."""
    s = _spikes({5: 0.99, 15: 0.90, 25: 0.80, 35: 0.70, 45: 0.60})
    e = _spikes({10: 0.99, 20: 0.90, 30: 0.80, 40: 0.70, 50: 0.60})
    counts = [len(decode_pairdp(np.zeros(N), s, e, TS, _cfg(pair_penalty=lam)))
              for lam in (0.0, 2.0, 4.0, 6.0, 20.0)]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] >= 4 and counts[-1] == 0, counts


def test_penalty_keeps_the_strongest_segments_first():
    s = _spikes({5: 0.99, 25: 0.60})
    e = _spikes({10: 0.99, 30: 0.60})
    kept = decode_pairdp(np.zeros(N), s, e, TS, _cfg(pair_penalty=1.5))
    assert kept == [(5.0, 10.0)], kept
