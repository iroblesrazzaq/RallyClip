from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class PlayerAssigner:
    screen_width: int = 1280
    screen_height: int = 720
    merge_iou_thresh: float = 0.6

    def __post_init__(self) -> None:
        self.screen_center_x = self.screen_width / 2
        self.left_zone_x = self.screen_width * 0.10
        self.right_zone_x = self.screen_width * 0.90
        self.bottom_zone_y = self.screen_height * 0.80

    def _calculate_iou(self, box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _conditional_merge_boxes(self, boxes, keypoints, confs, box_conf):
        if len(boxes) <= 1:
            return boxes, keypoints, confs, box_conf
        detections = [
            {
                "box": boxes[i],
                "keypoints": keypoints[i],
                "conf": confs[i],
                "box_conf": box_conf[i],
                "clump_id": -1,
            }
            for i in range(len(boxes))
        ]
        clump_count = 0
        for i in range(len(detections)):
            if detections[i]["clump_id"] == -1:
                detections[i]["clump_id"] = clump_count
                for j in range(i + 1, len(detections)):
                    if self._calculate_iou(detections[i]["box"], detections[j]["box"]) > self.merge_iou_thresh:
                        detections[j]["clump_id"] = clump_count
                clump_count += 1
        if clump_count == len(detections):
            return boxes, keypoints, confs, box_conf

        final_boxes, final_keypoints, final_confs, final_box_conf = [], [], [], []
        for clump_id in range(clump_count):
            clump = [d for d in detections if d["clump_id"] == clump_id]
            if len(clump) == 1:
                final_boxes.append(clump[0]["box"])
                final_keypoints.append(clump[0]["keypoints"])
                final_confs.append(clump[0]["conf"])
                final_box_conf.append(clump[0]["box_conf"])
                continue
            min_x1 = min(d["box"][0] for d in clump)
            min_y1 = min(d["box"][1] for d in clump)
            max_x2 = max(d["box"][2] for d in clump)
            max_y2 = max(d["box"][3] for d in clump)
            clump_center_x = (min_x1 + max_x2) / 2
            is_in_edge_zone = (
                clump_center_x < self.left_zone_x
                or clump_center_x > self.right_zone_x
                or max_y2 > self.bottom_zone_y
            )
            if is_in_edge_zone:
                merged_box = [min_x1, min_y1, max_x2, max_y2]
                best_detection = max(
                    clump, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
                )
                final_boxes.append(merged_box)
                final_keypoints.append(best_detection["keypoints"])
                final_confs.append(best_detection["conf"])
                final_box_conf.append(best_detection["box_conf"])
            else:
                for d in clump:
                    final_boxes.append(d["box"])
                    final_keypoints.append(d["keypoints"])
                    final_confs.append(d["conf"])
                    final_box_conf.append(d["box_conf"])

        return (
            np.array(final_boxes),
            np.array(final_keypoints),
            np.array(final_confs),
            np.array(final_box_conf),
        )

    def assign(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        boxes = data["boxes"]
        keypoints = data["keypoints"]
        confs = data["keypoint_conf"]
        box_conf = data["box_conf"]

        if len(boxes) == 0:
            return _empty_players()

        boxes, keypoints, confs, box_conf = self._conditional_merge_boxes(boxes, keypoints, confs, box_conf)
        candidates = [
            {"box": boxes[i], "keypoints": keypoints[i], "conf": confs[i], "box_conf": box_conf[i]}
            for i in range(len(boxes))
        ]

        near_idx = max(range(len(candidates)), key=lambda i: candidates[i]["box"][3])
        near = candidates.pop(near_idx)
        far = None
        if candidates:
            far_idx = min(
                range(len(candidates)),
                key=lambda i: abs(((candidates[i]["box"][0] + candidates[i]["box"][2]) / 2) - self.screen_center_x),
            )
            far = candidates[far_idx]

        return _pack_players(near, far)


def _empty_players() -> Dict[str, np.ndarray]:
    return {
        "near_kps": np.full((1, 17, 2), -1.0, dtype=np.float32),
        "far_kps": np.full((1, 17, 2), -1.0, dtype=np.float32),
        "near_conf": np.full((1, 17), -1.0, dtype=np.float32),
        "far_conf": np.full((1, 17), -1.0, dtype=np.float32),
        "near_box": np.full((1, 4), -1.0, dtype=np.float32),
        "far_box": np.full((1, 4), -1.0, dtype=np.float32),
        "near_box_conf": np.full((1,), -1.0, dtype=np.float32),
        "far_box_conf": np.full((1,), -1.0, dtype=np.float32),
    }


def _pack_players(near: Dict, far: Dict | None) -> Dict[str, np.ndarray]:
    if far is None:
        far = {
            "keypoints": np.full((17, 2), -1.0, dtype=np.float32),
            "conf": np.full((17,), -1.0, dtype=np.float32),
            "box": np.full((4,), -1.0, dtype=np.float32),
            "box_conf": np.array(-1.0, dtype=np.float32),
        }

    return {
        "near_kps": np.asarray(near["keypoints"], dtype=np.float32)[None, ...],
        "far_kps": np.asarray(far["keypoints"], dtype=np.float32)[None, ...],
        "near_conf": np.asarray(near["conf"], dtype=np.float32)[None, ...],
        "far_conf": np.asarray(far["conf"], dtype=np.float32)[None, ...],
        "near_box": np.asarray(near["box"], dtype=np.float32)[None, ...],
        "far_box": np.asarray(far["box"], dtype=np.float32)[None, ...],
        "near_box_conf": np.asarray(near["box_conf"], dtype=np.float32)[None, ...],
        "far_box_conf": np.asarray(far["box_conf"], dtype=np.float32)[None, ...],
    }
