#!/usr/bin/env python3
"""Extract raw YOLO pose HDF5 files

Runs the pipeline's `extract` step, so paths and settings match train.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.pipeline import run_step_cli  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    return run_step_cli("extract", "Extract raw YOLO pose HDF5 files", argv)


if __name__ == "__main__":
    raise SystemExit(main())
