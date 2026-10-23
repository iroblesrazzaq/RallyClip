"""extract_crop_poses: the crop decode is fixed at FPS, so raw poses must be too."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("extract_crop_poses", _ROOT / "scripts" / "extract_crop_poses.py")
crop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crop)


def test_raw_poses_on_the_crop_grid_pass():
    crop.check_grid(np.arange(10) / crop.FPS)


def test_raw_poses_on_another_grid_are_refused():
    with pytest.raises(ValueError, match="fps grid"):
        crop.check_grid(np.arange(10) / (2 * crop.FPS))
