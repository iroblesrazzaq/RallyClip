"""Second pose pass on a fixed far-court crop.

A 1920-wide frame letterboxes to imgsz=960 at scale 0.5, which leaves the far
player at ~80px, near YOLO's floor for pose. Running the same model on the
top-centre 960x540 window of the 1920x1080 frame feeds them in at native scale.

Shared by scripts/extract_crop_poses.py (training side-car) and the runtime pose
extractor. The window is fractional so it applies to any source resolution;
on 1920x1080 it is exactly the training crop (480, 0, 1440, 540).
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

CROP_WINDOW = (0.25, 0.0, 0.75, 0.5)  # x1, y1, x2, y2 as fractions of the frame
CROP_OUT_SIZE = (960, 540)            # crop resized to this (w, h): half of 1920x1080
MERGE_IOU = 0.6                       # matches PlayerAssigner.merge_iou_thresh


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_detections(boxes, box_conf, kps, kp_conf, thresh=MERGE_IOU):
    """Union detections that overlap above `thresh` (single-linkage).

    A player clipped by the crop boundary routinely yields several partial boxes
    that survive YOLO's own NMS; without this they consume both crop slots and
    evict the far player. The union box is kept, along with the keypoints of the
    highest-confidence member of the cluster.
    """
    n = len(boxes)
    if n <= 1:
        return boxes, box_conf, kps, kp_conf
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(boxes[i], boxes[j]) > thresh:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    ob, oc, ok, okc = [], [], [], []
    for members in groups.values():
        bs = boxes[members]
        best = members[int(np.argmax(box_conf[members]))]
        ob.append([bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()])
        oc.append(float(box_conf[best]))
        ok.append(kps[best])
        okc.append(kp_conf[best])
    order = np.argsort(-np.asarray(oc))
    return (np.asarray(ob, np.float32)[order], np.asarray(oc, np.float32)[order],
            np.asarray(ok, np.float32)[order], np.asarray(okc, np.float32)[order])


def crop_pixels(width: int, height: int, window: Sequence[float] = CROP_WINDOW) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = window
    return (int(round(x1 * width)), int(round(y1 * height)), int(round(x2 * width)), int(round(y2 * height)))


def crop_frame(frame_bgr: np.ndarray, window: Sequence[float] = CROP_WINDOW,
               out_size: tuple[int, int] = CROP_OUT_SIZE) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = crop_pixels(w, h, window)
    crop = frame_bgr[y1:y2, x1:x2]
    if (crop.shape[1], crop.shape[0]) == tuple(out_size):
        return np.ascontiguousarray(crop)
    interp = cv2.INTER_AREA if crop.shape[1] > out_size[0] else cv2.INTER_LINEAR
    return cv2.resize(crop, out_size, interpolation=interp)


def crop_to_frame(boxes: np.ndarray, kps: np.ndarray, width: int, height: int,
                  window: Sequence[float] = CROP_WINDOW,
                  out_size: tuple[int, int] = CROP_OUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Map crop-pixel boxes [N, 4] and keypoints [N, K, 2] back to frame pixels."""
    x1, y1, x2, y2 = crop_pixels(width, height, window)
    sx, sy = (x2 - x1) / out_size[0], (y2 - y1) / out_size[1]
    boxes = boxes * np.array([sx, sy, sx, sy], np.float32) + np.array([x1, y1, x1, y1], np.float32)
    kps = kps * np.array([sx, sy], np.float32) + np.array([x1, y1], np.float32)
    return boxes.astype(np.float32), kps.astype(np.float32)
