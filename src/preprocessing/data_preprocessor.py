import os
import numpy as np

from preprocessing.court_detector import CourtDetector


class DataPreprocessor:
    def __init__(self, screen_width: int = 1280, screen_height: int = 720, merge_iou_thresh: float = 0.6, save_court_masks: bool = False) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width / 2
        self.merge_iou_thresh = merge_iou_thresh
        self.save_court_masks = save_court_masks
        self.left_zone_x = screen_width * 0.10
        self.right_zone_x = screen_width * 0.90
        self.bottom_zone_y = screen_height * 0.80

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

    def _conditional_merge_boxes(self, boxes, keypoints, confs):
        if len(boxes) <= 1:
            return boxes, keypoints, confs
        detections = [{'box': boxes[i], 'keypoints': keypoints[i], 'conf': confs[i], 'clump_id': -1} for i in range(len(boxes))]
        clump_count = 0
        for i in range(len(detections)):
            if detections[i]['clump_id'] == -1:
                detections[i]['clump_id'] = clump_count
                for j in range(i + 1, len(detections)):
                    if self._calculate_iou(detections[i]['box'], detections[j]['box']) > self.merge_iou_thresh:
                        detections[j]['clump_id'] = clump_count
                clump_count += 1
        if clump_count == len(detections):
            return boxes, keypoints, confs
        final_boxes, final_keypoints, final_confs = [], [], []
        for clump_id in range(clump_count):
            clump = [d for d in detections if d['clump_id'] == clump_id]
            if len(clump) == 1:
                final_boxes.append(clump[0]['box'])
                final_keypoints.append(clump[0]['keypoints'])
                final_confs.append(clump[0]['conf'])
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
            else:
                for d in clump:
                    final_boxes.append(d['box'])
                    final_keypoints.append(d['keypoints'])
                    final_confs.append(d['conf'])
        return np.array(final_boxes), np.array(final_keypoints), np.array(final_confs)

    def assign_players(self, frame_data):
        boxes = frame_data.get('boxes', np.array([]))
        keypoints = frame_data.get('keypoints', np.array([]))
        confs = frame_data.get('conf', np.array([]))
        clean_boxes, clean_keypoints, clean_confs = self._conditional_merge_boxes(boxes, keypoints, confs)
        assigned_players = {'near_player': None, 'far_player': None}
        if len(clean_boxes) == 0:
            return assigned_players
        candidates = [{'box': clean_boxes[i], 'keypoints': clean_keypoints[i], 'conf': clean_confs[i]} for i in range(len(clean_boxes))]
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
            detector = CourtDetector(yolo_model_path='models/yolov8s.pt', weights_dir='models')
            mask, clean_frame, metadata = detector.process_video(video_path, target_time=60)
            return mask
        except Exception as e:
            print(f"  ⚠️  Error during court detection: {e}")
            return None

    def filter_frame_by_court(self, frame_data, mask):
        if mask is None:
            return frame_data
        boxes = frame_data['boxes']
        keypoints = frame_data['keypoints']
        conf = frame_data['conf']
        kept_boxes, kept_keypoints, kept_conf = [], [], []
        for i, box in enumerate(boxes):
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            if (0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and mask[int(center_y), int(center_x)] == 0):
                kept_boxes.append(box)
                kept_keypoints.append(keypoints[i])
                kept_conf.append(conf[i])
        return { 'boxes': np.array(kept_boxes), 'keypoints': np.array(kept_keypoints), 'conf': np.array(kept_conf) }

    def preprocess_single_video(self, input_npz_path: str, video_path: str, output_npz_path: str, overwrite: bool = False) -> bool:
        try:
            if os.path.exists(output_npz_path) and not overwrite:
                print(f"  ✓ Already exists, skipping: {os.path.basename(output_npz_path)}")
                return True
            print(f"  Loading pose data from: {input_npz_path}")
            pose_data = np.load(input_npz_path, allow_pickle=True)['frames']
            print(f"  Loaded {len(pose_data)} frames")
            print(f"  Generating court mask from: {video_path}")
            mask = self.generate_court_mask(video_path)
            if mask is not None:
                print(f"  ✓ Generated mask: {mask.shape}")
            else:
                print(f"  ⚠️  No court mask available - processing without filtering")
            all_frame_data, all_targets, all_near_players, all_far_players = [], [], [], []
            for frame_idx, frame_data in enumerate(pose_data):
                annotation_status = frame_data.get('annotation_status', 0)
                all_targets.append(annotation_status)
                if annotation_status == -100:
                    all_frame_data.append({'boxes': np.array([]), 'keypoints': np.array([]), 'conf': np.array([])})
                    all_near_players.append(None)
                    all_far_players.append(None)
                    continue
                filtered_frame_data = self.filter_frame_by_court(frame_data, mask)
                all_frame_data.append(filtered_frame_data)
                assigned_players = self.assign_players(filtered_frame_data)
                all_near_players.append(assigned_players['near_player'])
                all_far_players.append(assigned_players['far_player'])
                if (frame_idx + 1) % 100 == 0:
                    print(f"    Processed {frame_idx + 1}/{len(pose_data)} frames")
            os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
            save_data = { 'frames': all_frame_data, 'targets': np.array(all_targets), 'near_players': all_near_players, 'far_players': all_far_players }
            if self.save_court_masks and mask is not None:
                save_data['court_mask'] = mask
            np.savez_compressed(output_npz_path, **save_data)
            print(f"  ✓ Preprocessed data saved to: {output_npz_path}")
            return True
        except Exception as e:
            print(f"  ❌ Error processing {input_npz_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
