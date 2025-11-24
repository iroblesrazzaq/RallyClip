import os
import csv
from typing import Optional, List, Tuple, Callable

import numpy as np
import torch
import scipy.ndimage
import joblib

from .model import TennisPointLSTM


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


