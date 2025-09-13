# %% [markdown]
# ### notebook to perform postprocessing and evaluate how good model actually is at marking start/stops

# %%
# imports
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

from tennis_dataset import TennisDataset
from lstm_model_arch import TennisPointLSTM


def gaussian_kernel1d(sigma: float, kernel_size: int) -> np.ndarray:
    """Create a 1D Gaussian kernel with given sigma and odd kernel_size."""
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    half = kernel_size // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_smooth(values: np.ndarray, sigma: float = 1.5, kernel_size: Optional[int] = None) -> np.ndarray:
    """Smooth a 1D signal using a Gaussian kernel via convolution."""
    if kernel_size is None:
        # Common rule of thumb: cover ~3 sigmas on each side
        kernel_size = int(max(3, 6 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
    kernel = gaussian_kernel1d(sigma=sigma, kernel_size=kernel_size)
    return np.convolve(values, kernel, mode='same')


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


def plot_probs_vs_target(
    probs: np.ndarray,
    target: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    """Plot probabilities, target (0/1), and optional predicted mask."""
    x = np.arange(len(probs))
    plt.figure(figsize=(12, 4))
    plt.plot(x, probs, label='probability', color='C0', linewidth=2)
    plt.step(x, target.astype(float), where='post', label='target', color='C2', alpha=0.7)
    if pred_mask is not None:
        plt.step(x, pred_mask.astype(float), where='post', label='prediction', color='C3', alpha=0.9)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frame')
    plt.ylabel('value')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def compute_frame_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute simple frame-level metrics: accuracy, precision, recall, F1."""
    t = target.astype(np.int32).ravel()
    p = pred.astype(np.int32).ravel()
    assert t.shape == p.shape, "target and pred shapes must match"

    tp = int(((t == 1) & (p == 1)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())

    total = max(1, t.size)
    acc = (tp + tn) / total
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-12, (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }

# %%
# load best model pth from checkpoints dir
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    best = checkpoint_dir / 'best_model.pth'
    if best.exists():
        return best
    # fallback: pick highest epoch checkpoint
    candidates = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if not candidates:
        return None
    def _epoch_num(p: Path) -> int:
        try:
            return int(p.stem.split('_')[-1])
        except Exception:
            return -1
    candidates.sort(key=_epoch_num)
    return candidates[-1]


def load_model_from_checkpoint(
    checkpoint_path: Path,
    input_size: int = 360,
    hidden_size: int = 128,
    num_layers: int = 2,
    bidirectional: bool = True,
    return_logits: bool = False,
) -> TennisPointLSTM:
    model = TennisPointLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2,
        bidirectional=bidirectional,
        return_logits=return_logits,
    )
    state = torch.load(str(checkpoint_path), map_location=device)
    result = model.load_state_dict(state, strict=False)
    if getattr(result, 'missing_keys', []) or getattr(result, 'unexpected_keys', []):
        print('Warning: checkpoint keys mismatch:')
        print(f"  missing: {getattr(result, 'missing_keys', [])}")
        print(f"  unexpected: {getattr(result, 'unexpected_keys', [])}")
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


CHECKPOINT_DIR = Path('checkpoints')
_ckpt = _find_best_checkpoint(CHECKPOINT_DIR)
assert _ckpt is not None, f"No checkpoint found in {CHECKPOINT_DIR}"
model = load_model_from_checkpoint(_ckpt)

# %%
# perform inference on test set: currently, only for single sequences of 150
TEST_H5 = Path('data/test.h5')
assert TEST_H5.exists(), f"Missing test set file: {TEST_H5}"

test_dataset = TennisDataset(str(TEST_H5))
num_sequences = len(test_dataset)
print(f"Test sequences available: {num_sequences}")

# choose a sequence index to visualize
seq_idx = 0  # change this as needed
assert 0 <= seq_idx < num_sequences, "seq_idx out of range"

seq_features, seq_target = test_dataset[seq_idx]  # shapes: (150, 360), (150,)
with torch.no_grad():
    inp = seq_features.unsqueeze(0).to(device)  # (1, 150, 360)
    out = model(inp)  # (1, 150, 1), sigmoid probs by default
raw_probs = out.squeeze().detach().cpu().numpy()  # (150,)
target_seq = seq_target.detach().cpu().numpy().astype(np.int32)  # (150,)
print(f"Seq {seq_idx}: probs shape={raw_probs.shape}, target shape={target_seq.shape}")

# %%
# look at plot of raw probabilities vs target for a single sequence - make function to plot this
plot_probs_vs_target(raw_probs, target_seq, title=f"Raw probabilities vs target (seq {seq_idx})")

# %%
# apply postprocessing steps, tune hyperparameters:
# gaussian smoothing
gaussian_sigma = 1.5
gaussian_kernel_size = None  # auto from sigma
smoothed_probs = gaussian_smooth(raw_probs, sigma=gaussian_sigma, kernel_size=gaussian_kernel_size)
print(f"Applied Gaussian smoothing: sigma={gaussian_sigma}, kernel_size={gaussian_kernel_size}")

# %%
# look at gaussian smoothing plot for a single sequence - use same plotting function for this
plot_probs_vs_target(smoothed_probs, target_seq, title=f"Smoothed probabilities vs target (seq {seq_idx})")


# %%
# hysteresis thresholding
# look at hysteresis thresholding plot for a single sequence - use same plotting function for this
low_thresh = 0.35
high_thresh = 0.65
min_dur_frames = 6  # suppress very short bursts

# Apply on smoothed probabilities
hyst_pred = hysteresis_threshold(smoothed_probs, low=low_thresh, high=high_thresh, min_duration=min_dur_frames)

metrics = compute_frame_metrics(target_seq, hyst_pred)
print("Frame metrics (hysteresis on smoothed):")
for k in ['accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn']:
    print(f"  {k}: {metrics[k]:.4f}" if isinstance(metrics[k], float) else f"  {k}: {metrics[k]}")

plot_probs_vs_target(
    smoothed_probs,
    target_seq,
    pred_mask=hyst_pred,
    title=(
        f"Smoothed probs + hysteresis pred (seq {seq_idx})\n"
        f"low={low_thresh}, high={high_thresh}, min_dur={min_dur_frames}, sigma={gaussian_sigma}"
    ),
)


