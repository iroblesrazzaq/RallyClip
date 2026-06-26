from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infer.inference import (  # noqa: E402
    generate_start_indices,
    run_windowed_inference_average_stream,
)


def _batch_reference(features: np.ndarray, run_window, sequence_length: int, overlap: int) -> np.ndarray:
    """Replicates the batch windowed-average accumulation (float32) for comparison."""
    num_frames = features.shape[0]
    starts = generate_start_indices(num_frames, sequence_length, overlap)
    summed = np.zeros(num_frames, dtype=np.float32)
    counts = np.zeros(num_frames, dtype=np.int32)
    for s in starts:
        probs = np.asarray(run_window(features[s:s + sequence_length].astype(np.float32)), dtype=np.float32)
        summed[s:s + sequence_length] += probs
        counts[s:s + sequence_length] += 1
    return np.divide(summed, np.maximum(counts, 1), dtype=np.float32)


def _make_run_window():
    # Deterministic, content-dependent per-window probabilities so the comparison is
    # sensitive to any window mis-indexing or accumulation-order change.
    def run_window(window: np.ndarray) -> np.ndarray:
        return (np.tanh(window.sum(axis=1)) * 0.5 + 0.5).astype(np.float32)

    return run_window


# (num_frames, sequence_length, overlap): regular, tail-distinct, exact-fit, single-window,
# tail-by-one, zero-overlap.
@pytest.mark.parametrize(
    "num_frames,sequence_length,overlap",
    [
        (250, 100, 50),   # regular, no trailing window
        (260, 100, 50),   # trailing end-anchored window (distinct)
        (200, 100, 50),   # exact fit, no trailing
        (100, 100, 50),   # single window
        (101, 100, 0),    # zero overlap + 1-frame tail
        (333, 64, 16),    # odd sizes
    ],
)
def test_stream_matches_batch_bit_for_bit(num_frames, sequence_length, overlap):
    rng = np.random.default_rng(0)
    features = rng.standard_normal((num_frames, 12)).astype(np.float32)
    run_window = _make_run_window()

    expected = _batch_reference(features, run_window, sequence_length, overlap)
    got = run_windowed_inference_average_stream(
        iter(features), run_window, sequence_length, overlap
    )

    assert got.shape == expected.shape
    assert got.dtype == np.float32
    # Bit-for-bit: identical operands accumulated in the same float32 order.
    assert np.array_equal(got, expected)


def test_stream_rejects_too_short_input():
    rng = np.random.default_rng(1)
    features = rng.standard_normal((30, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        run_windowed_inference_average_stream(iter(features), _make_run_window(), 100, 50)


def test_stream_validates_overlap():
    rng = np.random.default_rng(2)
    features = rng.standard_normal((200, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        run_windowed_inference_average_stream(iter(features), _make_run_window(), 100, 100)
