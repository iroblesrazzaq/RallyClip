from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def draw_text(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
