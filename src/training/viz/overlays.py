from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

COCO_SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, confs: np.ndarray, color: Tuple[int, int, int]) -> None:
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_keypoints(frame: np.ndarray, keypoints: np.ndarray, confs: np.ndarray, color: Tuple[int, int, int]) -> None:
    for kp, conf in zip(keypoints, confs):
        x, y = int(kp[0]), int(kp[1])
        if conf < 0.05:
            continue
        cv2.circle(frame, (x, y), 2, color, -1)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, confs: np.ndarray, color: Tuple[int, int, int]) -> None:
    for i, j in COCO_SKELETON:
        if confs[i] < 0.05 or confs[j] < 0.05:
            continue
        p1 = (int(keypoints[i][0]), int(keypoints[i][1]))
        p2 = (int(keypoints[j][0]), int(keypoints[j][1]))
        cv2.line(frame, p1, p2, color, 1)


def draw_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.3, color: Tuple[int, int, int] = (0, 0, 255)) -> None:
    if mask is None:
        return
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = frame.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_lines(frame: np.ndarray, lines: Iterable[Tuple[float, float, float, float]], color: Tuple[int, int, int], thickness: int = 2) -> None:
    for x1, y1, x2, y2 in lines:
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_text_block(frame: np.ndarray, lines: Iterable[str], origin: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 16
