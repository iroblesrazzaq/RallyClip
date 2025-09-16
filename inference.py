# file to run inference on an entire feature npz file, then apply postprocessing steps
# to output final start_time,end_time csv file
# %%
import os
import csv
import argparse
import numpy as np
import torch
from lstm_model_arch import TennisPointLSTM
import scipy.ndimage
from typing import Optional, List, Tuple
import joblib



GAUSSIAN_SIGMA = 1.5  # for smoothing



def load_model_from_checkpoint(
    checkpoint_path: str,
    input_size: int = 360,
    hidden_size: int = 128,
    num_layers: int = 2,
    bidirectional: bool = True,
    return_logits: bool = False,
):
    """Load model weights from checkpoint, adapting architecture if needed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Extract model state dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('lstm.') or k.startswith('fc.') for k in ckpt.keys()):
        state_dict = ckpt
    else:
        # Fallback: attempt to use as state_dict
        state_dict = ckpt

    # Infer architecture from weights if possible
    inferred_input_size = input_size
    inferred_hidden_size = hidden_size
    inferred_num_layers = num_layers
    inferred_bidirectional = bidirectional

    try:
        # weight_ih_l0 shape: (4*hidden_size, input_size)
        w_ih_l0 = state_dict.get('lstm.weight_ih_l0', None)
        if w_ih_l0 is not None:
            inferred_hidden_size = w_ih_l0.shape[0] // 4
            inferred_input_size = w_ih_l0.shape[1]

        # Determine num_layers by counting layers
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

        # Bidirectionality: presence of any reverse weights
        inferred_bidirectional = any('_reverse' in k for k in state_dict.keys())
    except Exception:
        pass

    # Build model with inferred architecture
    model = TennisPointLSTM(
        input_size=inferred_input_size,
        hidden_size=inferred_hidden_size,
        num_layers=inferred_num_layers,
        dropout=0.2,
        bidirectional=inferred_bidirectional,
        return_logits=return_logits,
    )

    # Load strictly now that shapes should match
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    print(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(input_size={inferred_input_size}, hidden_size={inferred_hidden_size}, "
        f"num_layers={inferred_num_layers}, bidirectional={inferred_bidirectional})"
    )
    return model, device

def hysteresis_threshold(
    values: np.ndarray,
    low: float = 0.3,
    high: float = 0.7,
    min_duration: int = 0,
) -> np.ndarray:
    """Apply 1D hysteresis thresholding to a probability-like signal.

    - Enter active state when values >= high
    - Exit active state when values < low
    - Optional min_duration suppresses short active segments
    Returns a 0/1 array of the same length.
    """
    assert 0.0 <= low < high <= 1.0, "Require 0 <= low < high <= 1"
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

    # Handle active segment reaching the end
    if active and start_idx is not None:
        end_idx = n
        if (end_idx - start_idx) >= max(0, min_duration):
            pred[start_idx:end_idx] = 1

    return pred.astype(np.int32)

def generate_start_indices(num_frames: int, sequence_length: int, overlap: int) -> List[int]:
    """Generate sequence start indices with specified overlap, covering all frames."""
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
) -> np.ndarray:
    """Run sliding-window inference and average overlapping predictions per frame."""
    num_frames = features.shape[0]
    start_indices = generate_start_indices(num_frames, sequence_length, overlap)
    print(f"Generated {len(start_indices)} sequences for {num_frames} frames (seq_len={sequence_length}, overlap={overlap})")

    summed_probs = np.zeros(num_frames, dtype=np.float32)
    counts = np.zeros(num_frames, dtype=np.int32)

    for seq_idx, start in enumerate(start_indices):
        seq_np = features[start:start + sequence_length, :].astype(np.float32)
        seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(seq_tensor)
        output_sequence = output_tensor.squeeze().detach().cpu().numpy().astype(np.float32)  # (seq_len,)

        summed_probs[start:start + sequence_length] += output_sequence
        counts[start:start + sequence_length] += 1

        if seq_idx < 3 or seq_idx >= len(start_indices) - 3:
            print(f"  Seq {seq_idx}: frames {start}-{start + sequence_length - 1}")

    if np.any(counts == 0):
        zeros = int(np.sum(counts == 0))
        print(f"WARNING: {zeros} frames not covered by any window; filling with zeros")

    avg_probs = np.divide(summed_probs, np.maximum(counts, 1), dtype=np.float32)
    return avg_probs


def extract_segments_from_binary(pred: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) where pred == 1."""
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


def write_segments_csv(
    segments: List[Tuple[int, int]],
    output_csv_path: str,
    fps: float,
    overwrite: bool = False,
) -> None:
    """Write segments to CSV with start_time,end_time in seconds."""
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
    print(f"✓ Wrote segments CSV: {output_csv_path} ({len(segments)} segments)")


def main():
    parser = argparse.ArgumentParser(description="Single-video inference: average windows, smooth, hysteresis, write CSV")
    parser.add_argument("--features", type=str, required=True, help="Path to input *_features.npz")
    parser.add_argument(f"--model-path", type=str, required=True, help="Path to LSTM .pth file")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV (start_time,end_time)")
    parser.add_argument("--scaler-path", type=str, required=True, help="required path to StandardScaler .joblib")

    parser.add_argument("--fps", type=float, default=15.0, help="Sampling FPS used during feature creation")
    parser.add_argument("--seq-len", type=int, default=300, help="Sequence length for inference windows")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap (frames) between windows")

    parser.add_argument("--sigma", type=float, default=GAUSSIAN_SIGMA, help="Gaussian smoothing sigma")
    parser.add_argument("--low", type=float, default=0.45, help="Hysteresis low threshold")
    parser.add_argument("--high", type=float, default=0.8, help="Hysteresis high threshold")
    parser.add_argument("--min-dur-sec", type=float, default=0.5, help="Minimum segment duration in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if it exists")

    args = parser.parse_args()

    # Load model
    model, device = load_model_from_checkpoint(
        args.model_path,
        bidirectional=True,
        return_logits=False,
    )

    # Load features
    data = np.load(args.features)
    features = data["features"]  # shape: (num_frames, input_size)
    num_frames = features.shape[0]
    print(f"Loaded features: {args.features} (frames={num_frames}, dim={features.shape[1]})")

    # Optional: normalize with trained scaler
    if args.scaler_path is not None and os.path.exists(args.scaler_path):
        try:
            scaler = joblib.load(args.scaler_path)
            # Expect features shape (num_frames, dim) → transform per-frame
            features = scaler.transform(features)
            print(f"Applied scaler from {args.scaler_path}")
        except Exception as e:
            print(f"WARNING: Failed to load/apply scaler {args.scaler_path}: {e}")

    # Windowed inference with averaging
    avg_probs = run_windowed_inference_average(
        model=model,
        device=device,
        features=features,
        sequence_length=args.seq_len,
        overlap=args.overlap,
    )

    # Smoothing
    smoothed_probs = scipy.ndimage.gaussian_filter1d(avg_probs.astype(np.float32), sigma=float(args.sigma))
    print(
        "smoothed stats:",
        "min=", float(np.nanmin(smoothed_probs)),
        "max=", float(np.nanmax(smoothed_probs)),
        "nans=", int(np.isnan(smoothed_probs).sum()),
    )

    # Hysteresis
    min_duration_frames = int(round(max(0.0, args.min_dur_sec) * args.fps))
    binary_pred = hysteresis_threshold(smoothed_probs, low=float(args.low), high=float(args.high), min_duration=min_duration_frames)

    # Extract segments and write CSV in seconds (0-based timeline)
    segments = extract_segments_from_binary(binary_pred)
    write_segments_csv(segments, args.output, fps=float(args.fps), overwrite=bool(args.overwrite))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
