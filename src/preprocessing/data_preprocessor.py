import os
import logging
from pathlib import Path

import numpy as np

from preprocessing.court_detector import CourtDetector
from runtime.video_validation import probe_video


def rescale_player_to_reference(player, src_width, src_height, ref_width, ref_height):
    """Map a player's box + keypoints from native source pixels into the model's
    reference resolution (ref_width x ref_height).

    The v1 features are raw-pixel and the model was trained only at the reference
    resolution (1280x720), so non-reference input (e.g. 1080p/4K) must be linearly
    rescaled to match the training distribution. Identity when source == reference,
    so 720p input is unchanged. Returns a new player dict (or None)."""
    if player is None:
        return None
    if src_width <= 0 or src_height <= 0:
        return player
    sx = ref_width / src_width
    sy = ref_height / src_height
    if sx == 1.0 and sy == 1.0:
        return player
    box = np.asarray(player["box"], dtype=np.float32).copy()
    box[0] *= sx
    box[1] *= sy
    box[2] *= sx
    box[3] *= sy
    keypoints = np.asarray(player["keypoints"], dtype=np.float32).copy()
    keypoints[..., 0] *= sx
    keypoints[..., 1] *= sy
    return {**player, "box": box, "keypoints": keypoints}


class DataPreprocessor:
    def __init__(self, screen_width: int = 1280, screen_height: int = 720, merge_iou_thresh: float = 0.6, save_court_masks: bool = False, yolo_model_path: str = "yolov8n-pose.pt", conf: float = 0.25, yolo_device: str | None = None) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width / 2
        self.merge_iou_thresh = merge_iou_thresh
        self.save_court_masks = save_court_masks
        # Court detection uses the SAME YOLO model + conf as pose extraction (the values
        # pinned in the model manifest) so there is one model in play and one auto-download.
        self.yolo_model_path = yolo_model_path
        self.conf = float(conf)
        self.yolo_device = yolo_device
        self.left_zone_x = screen_width * 0.10
        self.right_zone_x = screen_width * 0.90
        self.bottom_zone_y = screen_height * 0.80
        self._default_court_mask_cache = None

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
                'box': boxes[i],
                'keypoints': keypoints[i],
                'conf': confs[i],
                'box_conf': box_conf[i],
                'clump_id': -1,
            }
            for i in range(len(boxes))
        ]
        clump_count = 0
        for i in range(len(detections)):
            if detections[i]['clump_id'] == -1:
                detections[i]['clump_id'] = clump_count
                for j in range(i + 1, len(detections)):
                    if self._calculate_iou(detections[i]['box'], detections[j]['box']) > self.merge_iou_thresh:
                        detections[j]['clump_id'] = clump_count
                clump_count += 1
        if clump_count == len(detections):
            return boxes, keypoints, confs, box_conf
        final_boxes, final_keypoints, final_confs, final_box_conf = [], [], [], []
        for clump_id in range(clump_count):
            clump = [d for d in detections if d['clump_id'] == clump_id]
            if len(clump) == 1:
                final_boxes.append(clump[0]['box'])
                final_keypoints.append(clump[0]['keypoints'])
                final_confs.append(clump[0]['conf'])
                final_box_conf.append(clump[0]['box_conf'])
                continue
            min_x1 = min(d['box'][0] for d in clump)
            min_y1 = min(d['box'][1] for d in clump)
            max_x2 = max(d['box'][2] for d in clump)
            max_y2 = max(d['box'][3] for d in clump)
            clump_center_x = (min_x1 + max_x2) / 2
            is_in_edge_zone = (
                clump_center_x < self.left_zone_x or
                clump_center_x > self.right_zone_x or
                max_y2 > self.bottom_zone_y
            )
            if is_in_edge_zone:
                merged_box = [min_x1, min_y1, max_x2, max_y2]
                best_detection = max(clump, key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))
                final_boxes.append(merged_box)
                final_keypoints.append(best_detection['keypoints'])
                final_confs.append(best_detection['conf'])
                final_box_conf.append(best_detection['box_conf'])
            else:
                for d in clump:
                    final_boxes.append(d['box'])
                    final_keypoints.append(d['keypoints'])
                    final_confs.append(d['conf'])
                    final_box_conf.append(d['box_conf'])
        return np.array(final_boxes), np.array(final_keypoints), np.array(final_confs), np.array(final_box_conf)

    def assign_players(self, frame_data):
        boxes = np.asarray(frame_data.get('boxes', np.empty((0, 4), dtype=np.float32)), dtype=np.float32)
        keypoints = np.asarray(frame_data.get('keypoints', np.empty((0, 17, 2), dtype=np.float32)), dtype=np.float32)
        confs = np.asarray(frame_data.get('conf', np.empty((0, 17), dtype=np.float32)), dtype=np.float32)
        box_conf = np.asarray(frame_data.get('box_conf', np.full((len(boxes),), -1.0, dtype=np.float32)), dtype=np.float32)
        if box_conf.shape[0] != len(boxes):
            box_conf = np.full((len(boxes),), -1.0, dtype=np.float32)
        clean_boxes, clean_keypoints, clean_confs, clean_box_conf = self._conditional_merge_boxes(boxes, keypoints, confs, box_conf)
        assigned_players = {'near_player': None, 'far_player': None}
        if len(clean_boxes) == 0:
            return assigned_players
        candidates = [
            {
                'box': clean_boxes[i],
                'keypoints': clean_keypoints[i],
                'conf': clean_confs[i],
                'box_conf': clean_box_conf[i],
            }
            for i in range(len(clean_boxes))
        ]
        near_player_idx = max(range(len(candidates)), key=lambda i: candidates[i]['box'][3])
        near_player = candidates[near_player_idx]
        assigned_players['near_player'] = near_player
        del candidates[near_player_idx]
        if len(candidates) > 0:
            far_player_idx = min(range(len(candidates)), key=lambda i: abs(((candidates[i]['box'][0] + candidates[i]['box'][2]) / 2) - self.screen_center_x))
            far_player = candidates[far_player_idx]
            assigned_players['far_player'] = far_player
        return assigned_players

    def generate_court_mask(self, video_path: str):
        try:
            detector = CourtDetector(yolo_model_path=self.yolo_model_path, conf=self.conf, device=self.yolo_device)
            mask, clean_frame, metadata = detector.process_video(video_path, target_time=60)
            return mask
        except Exception as e:
            logging.warning("Court detection failed: %s", e)
            return None

    def _court_sample_times(self, video_path: str) -> list:
        """Timestamps (seconds) to try for court detection, spread across the video."""
        duration = 0.0
        try:
            duration = float(probe_video(video_path).duration_s)
        except Exception:
            duration = 0.0
        if duration > 10:
            return sorted({max(1, int(duration * f)) for f in (0.2, 0.35, 0.5, 0.65, 0.8)})
        return [60, 90, 45, 120, 30]

    def _load_default_court_mask(self, frame_shape) -> np.ndarray:
        """Load the empirical default 'out' mask, resized to the given (H, W, ...) frame shape."""
        import cv2
        if self._default_court_mask_cache is None:
            path = Path(__file__).resolve().parent / "default_court_mask.png"
            base = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if base is None:
                raise FileNotFoundError(f"Default court mask not found at {path}")
            # OpenCV 4.13 on some platforms returns (H, W, 1) for grayscale
            # reads; the mask contract is 2-D (H, W).
            self._default_court_mask_cache = base.reshape(base.shape[0], base.shape[1])
        base = self._default_court_mask_cache
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if base.shape[:2] != (h, w):
            base = cv2.resize(base, (w, h), interpolation=cv2.INTER_NEAREST)
        return (base > 127).astype(np.uint8) * 255

    def _source_frame_shape(self, video_path: str) -> tuple[int, int, int]:
        """Return native video shape as (height, width, channels), falling back to reference size."""
        try:
            info = probe_video(video_path)
            if info.width > 0 and info.height > 0:
                return (int(info.height), int(info.width), 3)
        except Exception as exc:
            logging.warning("Could not probe video dimensions for %s: %s", video_path, exc)
        return (self.screen_height, self.screen_width, 3)

    def compute_court_mask(self, video_path: str):
        """Front-loaded court detection with a default-mask fallback.

        Runs the detector at several timestamps and returns the first plausible 'out'
        mask. If detection fails at every sample, returns the empirical default mask
        (resized to the frame) so frames are still filtered rather than passed through
        unmasked. Always returns (mask, metadata) with a usable mask (never None).
        """
        detector = CourtDetector(yolo_model_path=self.yolo_model_path, conf=self.conf, device=self.yolo_device)
        frame_shape = self._source_frame_shape(video_path)
        for t in self._court_sample_times(video_path):
            try:
                mask, clean_frame, _meta = detector.process_video(video_path, target_time=t)
            except Exception as e:
                logging.warning("Court detection error at t=%ss: %s", t, e)
                continue
            if getattr(clean_frame, "shape", None) is not None:
                frame_shape = clean_frame.shape
            if mask is not None and np.any(mask):
                logging.info("Court detected at t=%ss", t)
                return mask, {"source": "detected", "detected": True, "timestamp_s": t}
        logging.warning("Court detection failed at all sampled timestamps; using default court mask")
        default = self._load_default_court_mask(frame_shape)
        return default, {"source": "default", "detected": False, "timestamp_s": None}

    def filter_frame_by_court(self, frame_data, mask):
        boxes = np.asarray(frame_data.get('boxes', np.empty((0, 4), dtype=np.float32)), dtype=np.float32)
        keypoints = np.asarray(frame_data.get('keypoints', np.empty((0, 17, 2), dtype=np.float32)), dtype=np.float32)
        conf = np.asarray(frame_data.get('conf', np.empty((0, 17), dtype=np.float32)), dtype=np.float32)
        box_conf = frame_data.get('box_conf', np.full((len(boxes),), -1.0, dtype=np.float32))
        box_conf = np.asarray(box_conf, dtype=np.float32)
        if box_conf.shape[0] != len(boxes):
            box_conf = np.full((len(boxes),), -1.0, dtype=np.float32)
        normalized = {
            'boxes': boxes.reshape((-1, 4)),
            'box_conf': box_conf,
            'keypoints': keypoints.reshape((-1, 17, 2)),
            'conf': conf.reshape((-1, 17)),
        }
        if mask is None:
            return normalized
        kept_boxes, kept_box_conf, kept_keypoints, kept_conf = [], [], [], []
        for i, box in enumerate(normalized['boxes']):
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            if (0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and mask[int(center_y), int(center_x)] == 0):
                kept_boxes.append(box)
                kept_box_conf.append(normalized['box_conf'][i])
                kept_keypoints.append(normalized['keypoints'][i])
                kept_conf.append(normalized['conf'][i])
        return {
            'boxes': np.asarray(kept_boxes, dtype=np.float32).reshape((-1, 4)),
            'box_conf': np.asarray(kept_box_conf, dtype=np.float32),
            'keypoints': np.asarray(kept_keypoints, dtype=np.float32).reshape((-1, 17, 2)),
            'conf': np.asarray(kept_conf, dtype=np.float32).reshape((-1, 17)),
        }

    def iter_preprocess_frames(self, pose_data, court_mask, src_width: int, src_height: int):
        """Streaming preprocessing core (generator).

        Consumes the pose stream and yields one ``(frame_data, target, near_player,
        far_player)`` record per frame, in order. Each frame is processed and released
        independently (court filter -> player assignment -> reference rescale), so peak
        memory is O(1) in video length. :meth:`preprocess_frames` materializes this into
        the parallel-list dict; the release feature path only needs the target/near/far
        fields, so it can consume the records directly without holding the full lists.
        """
        mask = court_mask
        for frame_idx, frame_data in enumerate(pose_data):
            annotation_status = frame_data.get('annotation_status', 0)
            if annotation_status == -1:
                yield (
                    {
                        'boxes': np.empty((0, 4), dtype=np.float32),
                        'box_conf': np.empty((0,), dtype=np.float32),
                        'keypoints': np.empty((0, 17, 2), dtype=np.float32),
                        'conf': np.empty((0, 17), dtype=np.float32),
                    },
                    annotation_status,
                    None,
                    None,
                )
                continue
            filtered_frame_data = self.filter_frame_by_court(frame_data, mask)
            assigned_players = self.assign_players(filtered_frame_data)
            near_player = rescale_player_to_reference(
                assigned_players['near_player'], src_width, src_height, self.screen_width, self.screen_height)
            far_player = rescale_player_to_reference(
                assigned_players['far_player'], src_width, src_height, self.screen_width, self.screen_height)
            if (frame_idx + 1) % 100 == 0:
                logging.debug("Processed %s frames", frame_idx + 1)
            yield (filtered_frame_data, annotation_status, near_player, far_player)

    def preprocess_frames(self, pose_data, court_mask, src_width: int, src_height: int) -> dict:
        """In-memory preprocessing core: the full parallel-list dict.

        Materializes :meth:`iter_preprocess_frames` into the dict the release path and
        :meth:`preprocess_single_video` consume:
        ``{'frames', 'targets', 'near_players', 'far_players'}``.
        """
        all_frame_data, all_targets, all_near_players, all_far_players = [], [], [], []
        for frame_data, target, near_player, far_player in self.iter_preprocess_frames(
            pose_data, court_mask, src_width, src_height
        ):
            all_frame_data.append(frame_data)
            all_targets.append(target)
            all_near_players.append(near_player)
            all_far_players.append(far_player)
        return {'frames': all_frame_data, 'targets': np.array(all_targets), 'near_players': all_near_players, 'far_players': all_far_players}

    def preprocess_single_video(self, input_npz_path: str, video_path: str, output_npz_path: str, overwrite: bool = False, court_mask=None) -> bool:
        """File-writing wrapper around :meth:`preprocess_frames` (training scripts).

        Behaviour-preserving load->core->save: loads pose data from NPZ, resolves the
        court mask + native resolution, runs the core, and serializes the same dict.
        """
        try:
            if os.path.exists(output_npz_path) and not overwrite:
                logging.info("Preprocess skip (exists): %s", os.path.basename(output_npz_path))
                return True
            logging.info("Loading pose data from: %s", input_npz_path)
            with np.load(input_npz_path, allow_pickle=True) as data:
                pose_data = data['frames'].copy()
            logging.info("Loaded %s frames", len(pose_data))
            if court_mask is None:
                logging.info("Generating court mask from: %s", video_path)
                mask = self.generate_court_mask(video_path)
            else:
                mask = court_mask
            if mask is not None:
                logging.info("Court mask: %s", getattr(mask, 'shape', None))
            else:
                logging.info("No court mask available - processing without filtering")
            # Native source resolution, independent of court-mask fallback shape.
            # Used to rescale detections into the model's reference space; identity at 720p.
            src_height, src_width, _ = self._source_frame_shape(video_path)
            save_data = self.preprocess_frames(pose_data, mask, src_width, src_height)
            os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
            if self.save_court_masks and mask is not None:
                save_data['court_mask'] = mask
            np.savez_compressed(output_npz_path, **save_data)
            logging.info("Preprocessed data saved: %s", output_npz_path)
            return True
        except Exception as e:
            logging.error("Error processing %s: %s", input_npz_path, e, exc_info=True)
            return False
