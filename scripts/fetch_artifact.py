#!/usr/bin/env python3
"""Download the shipped ONNX artifact into models/rallyclip_v0.5.0/.

Idempotent: skips the network when SHA-256 already matches SHA256SUMS.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from runtime.artifact import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["fetch", "--repo-root", str(_ROOT), *sys.argv[1:]]))
