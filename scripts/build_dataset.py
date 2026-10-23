#!/usr/bin/env python3
"""Build a training dataset from feature files

Runs the pipeline's `dataset` step, so paths and settings match train.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.pipeline import run_step_cli  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    return run_step_cli("dataset", "Build a training dataset from feature files", argv)


if __name__ == "__main__":
    raise SystemExit(main())
