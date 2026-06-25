import os
import csv
import json
from typing import Optional, List, Tuple, Callable

import numpy as np
import torch
import joblib

from .model import TennisPointLSTM

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional runtime dependency
    ort = None


class JsonStandardScaler:
    """Small scaler adapter for JSON-exported StandardScaler assets."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean_ = mean.astype(np.float32)
        self.scale_ = scale.astype(np.float32)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != self.mean_.shape[0]:
            raise ValueError(
                f"Scaler feature mismatch: got {values.shape[-1]}, expected {self.mean_.shape[0]}"
            )
        return (values - self.mean_) / np.maximum(self.scale_, 1e-12)


def load_scaler_asset(path: str):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        mean = np.asarray(payload.get("mean", []), dtype=np.float32)
        scale = np.asarray(payload.get("scale", []), dtype=np.float32)
        if mean.size == 0 or scale.size == 0:
            raise ValueError(f"Invalid scaler JSON: '{path}'")
        return JsonStandardScaler(mean=mean, scale=scale)
    return joblib.load(path)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_windowed_inference_average_onnx(
    model_path: str,
    features: np.ndarray,
    sequence_length: int,
    overlap: int,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    if ort is None:
        raise RuntimeError(
            "ONNX model requested but onnxruntime is not installed. "
            "Install dependencies with `pip install .`."
        )

    providers = [p for p in ort.get_available_providers() if p in {"CPUExecutionProvider", "CUDAExecutionProvider"}]
    if not providers:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    num_frames = features.shape[0]
    start_indices = generate_start_indices(num_frames, sequence_length, overlap)
    summed_probs = np.zeros(num_frames, dtype=np.float32)
    counts = np.zeros(num_frames, dtype=np.int32)

    for seq_idx, start in enumerate(start_indices):
        seq_np = features[start:start + sequence_length, :].astype(np.float32)[None, ...]
        output = session.run([output_name], {input_name: seq_np})[0]
        seq_out = np.asarray(output, dtype=np.float32).squeeze()
        if seq_out.ndim != 1:
            seq_out = seq_out.reshape(-1)
        probs = _sigmoid(seq_out)
        if probs.shape[0] != sequence_length:
            raise ValueError(
                f"ONNX output length mismatch: got {probs.shape[0]}, expected {sequence_length}. "
                "Check seq_len against model manifest."
            )
        summed_probs[start:start + sequence_length] += probs
        counts[start:start + sequence_length] += 1
        if progress_callback is not None:
            try:
                progress_callback((seq_idx + 1) / float(len(start_indices)))
            except Exception:
                pass

    return np.divide(summed_probs, np.maximum(counts, 1), dtype=np.float32)


def gaussian_filter1d(data: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian filter using numpy convolution (replaces scipy.ndimage.gaussian_filter1d)."""
    if sigma <= 0:
        return data.copy()
    radius = int(3 * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(data, radius, mode='edge')
    return np.convolve(padded, kernel, mode='valid').astype(data.dtype)


def load_model_from_checkpoint(
    checkpoint_path: str,
    input_size: int = 360,
    hidden_size: int = 128,
    num_layers: int = 2,
    bidirectional: bool = True,
    return_logits: bool = False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('lstm.') or k.startswith('fc.') for k in ckpt.keys()):
        state_dict = ckpt
    else:
        state_dict = ckpt

    inferred_input_size = input_size
    inferred_hidden_size = hidden_size
    inferred_num_layers = num_layers
    inferred_bidirectional = bidirectional
    try:
        w_ih_l0 = state_dict.get('lstm.weight_ih_l0', None)
        if w_ih_l0 is not None:
            inferred_hidden_size = w_ih_l0.shape[0] // 4
            inferred_input_size = w_ih_l0.shape[1]
        layer_indices = set()
        for k in state_dict.keys():
            if k.startswith('lstm.weight_ih_l'):
                try:
                    idx_str = k.split('lstm.weight_ih_l')[1]
                    idx = int(idx_str.split('_')[0]) if '_' in idx_str else int(idx_str)
                    layer_indices.add(idx)
                except Exception:
                    pass
        if layer_indices:
            inferred_num_layers = max(layer_indices) + 1
        inferred_bidirectional = any('_reverse' in k for k in state_dict.keys())
    except Exception:
        pass

    model = TennisPointLSTM(
        input_size=inferred_input_size,
        hidden_size=inferred_hidden_size,
        num_layers=inferred_num_layers,
        dropout=0.2,
        bidirectional=inferred_bidirectional,
        return_logits=return_logits,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device


def hysteresis_threshold(values: np.ndarray, low: float = 0.3, high: float = 0.7, min_duration: int = 0) -> np.ndarray:
    assert 0.0 <= low < high <= 1.0
    n = len(values)
    pred = np.zeros(n, dtype=np.int8)
    active = False
    start_idx: Optional[int] = None
    for i in range(n):
        v = values[i]
        if not active:
            if v >= high:
                active = True
                start_idx = i
        else:
            if v < low:
                end_idx = i
                if start_idx is not None and (end_idx - start_idx) >= max(0, min_duration):
                    pred[start_idx:end_idx] = 1
                active = False
                start_idx = None
    if active and start_idx is not None:
        end_idx = n
        if (end_idx - start_idx) >= max(0, min_duration):
            pred[start_idx:end_idx] = 1
    return pred.astype(np.int32)


def generate_start_indices(num_frames: int, sequence_length: int, overlap: int) -> List[int]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if overlap < 0 or overlap >= sequence_length:
        raise ValueError("overlap must be in [0, sequence_length-1]")
    if num_frames < sequence_length:
        raise ValueError("input video too short for the chosen sequence_length")
    step = sequence_length - overlap
    start_indices: List[int] = []
    idx = 0
    while idx + sequence_length <= num_frames:
        start_indices.append(idx)
        idx += step
    if start_indices[-1] + sequence_length < num_frames:
        start_indices.append(num_frames - sequence_length)
    return start_indices


def run_windowed_inference_average(
    model: TennisPointLSTM,
    device: torch.device,
    features: np.ndarray,
    sequence_length: int,
    overlap: int,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    num_frames = features.shape[0]
    start_indices = generate_start_indices(num_frames, sequence_length, overlap)
    summed_probs = np.zeros(num_frames, dtype=np.float32)
    counts = np.zeros(num_frames, dtype=np.int32)
    for seq_idx, start in enumerate(start_indices):
        seq_np = features[start:start + sequence_length, :].astype(np.float32)
        seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(seq_tensor)
        output_sequence = output_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
        summed_probs[start:start + sequence_length] += output_sequence
        counts[start:start + sequence_length] += 1
        if progress_callback is not None:
            try:
                progress_callback((seq_idx + 1) / float(len(start_indices)))
            except Exception:
                pass
    avg_probs = np.divide(summed_probs, np.maximum(counts, 1), dtype=np.float32)
    return avg_probs


def run_windowed_inference_average_stream(
    feature_rows,
    run_window: Callable[[np.ndarray], np.ndarray],
    sequence_length: int,
    overlap: int,
    progress_callback: Optional[Callable[[float], None]] = None,
    total_windows: Optional[int] = None,
) -> np.ndarray:
    """Windowed-average inference over a *stream* of feature rows (bounded memory).

    Produces the same averaged-probability vector as :func:`run_windowed_inference_average`
    / ``_onnx`` but holds only a ``sequence_length`` ring buffer of feature rows instead of
    the full ``(num_frames, F)`` matrix, so peak memory is O(seq_len) in the features rather
    than O(num_frames). The only O(num_frames) state is the float-per-frame ``summed`` /
    ``counts`` accumulators.

    Args:
        feature_rows: iterable of 1-D float32 feature vectors, one per frame, in order.
        run_window: maps a ``(sequence_length, F)`` window to its ``sequence_length``
            probabilities -- the same per-window computation the batch callers perform
            (torch forward, or ONNX run + sigmoid).
        sequence_length, overlap: identical windowing to :func:`generate_start_indices`.

    The window start sequence and the float32 slice-accumulation order are identical to the
    batch path, so the averaged output is bit-for-bit identical.
    """
    from collections import deque

    L = int(sequence_length)
    if L <= 0:
        raise ValueError("sequence_length must be > 0")
    if overlap < 0 or overlap >= L:
        raise ValueError("overlap must be in [0, sequence_length-1]")
    step = L - overlap

    ring: deque = deque(maxlen=L)
    cap = max(L, 1024)
    summed = np.zeros(cap, dtype=np.float32)
    counts = np.zeros(cap, dtype=np.int32)
    n = 0
    next_start = 0
    fired = 0

    def _grow(size: int) -> None:
        nonlocal summed, counts, cap
        if size <= cap:
            return
        new_cap = cap
        while new_cap < size:
            new_cap *= 2
        new_s = np.zeros(new_cap, dtype=np.float32)
        new_s[:summed.size] = summed
        new_c = np.zeros(new_cap, dtype=np.int32)
        new_c[:counts.size] = counts
        summed, counts, cap = new_s, new_c, new_cap

    def _fire(start: int) -> None:
        nonlocal fired
        window = np.stack(tuple(ring)).astype(np.float32)
        probs = np.asarray(run_window(window), dtype=np.float32)
        if probs.shape[0] != L:
            raise ValueError(
                f"window output length mismatch: got {probs.shape[0]}, expected {L}. "
                "Check seq_len against model manifest."
            )
        summed[start:start + L] += probs
        counts[start:start + L] += 1
        fired += 1
        if progress_callback is not None and total_windows:
            try:
                progress_callback(fired / float(total_windows))
            except Exception:
                pass

    for row in feature_rows:
        ring.append(np.asarray(row, dtype=np.float32))
        n += 1
        _grow(n)
        if next_start + L <= n:
            _fire(next_start)
            next_start += step

    if n < L:
        raise ValueError("input video too short for the chosen sequence_length")
    # End-anchored tail window, appended only when it does not coincide with the last
    # regular start -- matches generate_start_indices' trailing append exactly.
    if (next_start - step) + L < n:
        _fire(n - L)

    return np.divide(summed[:n], np.maximum(counts[:n], 1), dtype=np.float32)


def extract_segments_from_binary(pred: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    n = len(pred)
    if n == 0:
        return segments
    in_seg = False
    seg_start: Optional[int] = None
    for i in range(n):
        if not in_seg and pred[i] == 1:
            in_seg = True
            seg_start = i
        elif in_seg and pred[i] == 0:
            segments.append((seg_start, i))
            in_seg = False
            seg_start = None
    if in_seg and seg_start is not None:
        segments.append((seg_start, n))
    return segments


def write_segments_csv(segments: List[Tuple[int, int]], output_csv_path: str, fps: float, overwrite: bool = False) -> None:
    if os.path.exists(output_csv_path) and not overwrite:
        print(f"✓ Output exists, skipping write (set --overwrite to replace): {output_csv_path}")
        return
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "end_time"])  # header
        for start_idx, end_idx in segments:
            start_t = start_idx / fps
            end_t = end_idx / fps
            writer.writerow([f"{start_t:.3f}", f"{end_t:.3f}"])


