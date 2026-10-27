from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Training-only modules import torch during collection, before marker filters
# run. CPU desktop/CI installs deliberately omit the optional [train] extra.
# The full training environment still collects and runs these tests normally.
collect_ignore = [] if find_spec("torch") is not None else [
    "test_checkpoint_arch.py",
    "test_heatmap_loss.py",
    "test_heatmap_pairdp.py",
    "test_heatmap_tcn.py",
    "test_pos_weight.py",
    "test_train_head_dispatch.py",
    "test_train_smoke_both_heads.py",
]
