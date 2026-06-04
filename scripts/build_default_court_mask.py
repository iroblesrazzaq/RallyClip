"""Build the default court 'out' mask from the committed regression goldens.

Majority vote (a pixel is OUT when >= 50% of the courts mark it OUT) over
tests/fixtures/court/masks/*.png. This is the fallback used when live court
detection fails on a video; deriving it from the hand-annotated corpus keeps it
tracking the typical court framing. Re-run after the corpus changes.

Output: src/preprocessing/default_court_mask.png (1280x720, 0/255).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MASKS = REPO / "tests" / "fixtures" / "court" / "masks"
OUT = REPO / "src" / "preprocessing" / "default_court_mask.png"


def main() -> int:
    files = sorted(MASKS.glob("*.png"))
    if not files:
        raise SystemExit(f"No golden masks found in {MASKS}")
    stack = np.stack([cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 127 for f in files])
    majority = (stack.mean(axis=0) >= 0.5).astype(np.uint8) * 255
    cv2.imwrite(str(OUT), majority)
    out_pct = int((majority > 0).mean() * 100)
    print(f"Default court mask {majority.shape} (OUT={out_pct}% of frame) from {len(files)} courts -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
