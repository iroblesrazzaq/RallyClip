import os
import csv
import json
from typing import Optional, List, Tuple, Callable

import numpy as np
import torch
import joblib

from .model import TennisPointLSTM


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


def load_scaler_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_standard_scaler_json(features: np.ndarray, scaler_state: dict, eps: float = 1e-8) -> np.ndarray:
    feature_dim = int(scaler_state.get("feature_dim", 0) or 0)
    if feature_dim <= 0:
        raise ValueError("Scaler JSON must declare a positive feature_dim")
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array")
    if features.shape[1] != feature_dim:
        raise ValueError(
            f"Scaler feature dimension mismatch: expected {feature_dim}, got {features.shape[1]}"
        )

    mean = np.asarray(scaler_state.get("mean", []), dtype=np.float32)
    scale = np.asarray(scaler_state.get("scale", []), dtype=np.float32)
    if mean.shape != (feature_dim,) or scale.shape != (feature_dim,):
        raise ValueError("Scaler JSON mean/scale length must match feature_dim")

    safe_scale = np.where(np.abs(scale) < eps, 1.0, scale).astype(np.float32)
    return ((features.astype(np.float32) - mean) / safe_scale).astype(np.float32)


def create_onnx_session(model_path: str):
    import onnxruntime as ort

    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _normalize_onnx_window_output(output: np.ndarray, sequence_length: int) -> np.ndarray:
    arr = np.asarray(output, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Unexpected ONNX output shape after squeeze: {arr.shape}")
    if arr.shape[0] != sequence_length:
        raise ValueError(
            f"Unexpected ONNX output sequence length: expected {sequence_length}, got {arr.shape[0]}"
        )
    return arr.astype(np.float32)


def run_windowed_inference_average_onnx(
    session,
    features: np.ndarray,
    sequence_length: int,
    overlap: int,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array")

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = list(getattr(input_meta, "shape", []))
    if len(input_shape) >= 1 and isinstance(input_shape[0], int) and input_shape[0] not in (1,):
        raise ValueError(f"ONNX runtime only supports batch size 1, got model batch {input_shape[0]}")
    if len(input_shape) >= 2 and isinstance(input_shape[1], int) and input_shape[1] != sequence_length:
        raise ValueError(
            f"ONNX sequence length mismatch: expected {input_shape[1]}, got {sequence_length}"
        )
    if len(input_shape) >= 3 and isinstance(input_shape[2], int) and input_shape[2] != features.shape[1]:
        raise ValueError(
            f"ONNX feature dimension mismatch: expected {input_shape[2]}, got {features.shape[1]}"
        )

    num_frames = features.shape[0]
    start_indices = generate_start_indices(num_frames, sequence_length, overlap)
    summed_probs = np.zeros(num_frames, dtype=np.float32)
    counts = np.zeros(num_frames, dtype=np.int32)
    for seq_idx, start in enumerate(start_indices):
        seq_np = features[start:start + sequence_length, :].astype(np.float32)[None, :, :]
        output = session.run(None, {input_name: seq_np})
        output_sequence = _normalize_onnx_window_output(output[0], sequence_length)
        summed_probs[start:start + sequence_length] += output_sequence
        counts[start:start + sequence_length] += 1
        if progress_callback is not None:
            try:
                progress_callback((seq_idx + 1) / float(len(start_indices)))
            except Exception:
                pass
    return np.divide(summed_probs, np.maximum(counts, 1), dtype=np.float32)


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


def apply_postprocess(values: np.ndarray, method: str, params: dict, fps: float) -> np.ndarray:
    if method != "hysteresis":
        raise ValueError(f"Unknown postprocess method: {method}")
    sigma = float(params.get("sigma", 1.5))
    smoothed = gaussian_filter1d(values.astype(np.float32), sigma=sigma)
    min_duration_frames = int(round(max(0.0, float(params.get("min_dur_sec", 0.0))) * float(fps)))
    return hysteresis_threshold(
        smoothed,
        low=float(params.get("low", 0.45)),
        high=float(params.get("high", 0.8)),
        min_duration=min_duration_frames,
    )


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

