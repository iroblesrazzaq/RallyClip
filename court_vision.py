import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


try:
    # YOLO from ultralytics (pip install ultralytics)
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore

try:
    # Optional MoveNet via TensorFlow Hub (pip install tensorflow tensorflow-hub)
    import tensorflow as tf  # type: ignore
    import tensorflow_hub as hub  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore
    hub = None  # type: ignore


LABEL_TO_ID = {
    "INACTIVE": 0,
    "SERVE_MOTION": 1,
    "RALLY": 2,
}


class MoveNetWrapper:
    """Lightweight wrapper around MoveNet single-pose from TensorFlow Hub.

    If TensorFlow/TF-Hub are not available, this wrapper gracefully degrades by
    returning zeros for keypoints. This allows the rest of the pipeline to run
    for integration testing without pose features.
    """

    def __init__(self, model_name: str = "singlepose_lightning") -> None:
        self._ready = False
        self._input_size = 192
        self._model = None
        if tf is None or hub is None:
            return
        model_map = {
            "singlepose_lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
            "singlepose_thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
        }
        url = model_map.get(model_name, model_map["singlepose_lightning"])
        try:
            self._model = hub.load(url)
            self._ready = True
        except Exception:
            self._ready = False

    def infer_keypoints(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Run pose on a cropped RGB person image.

        Returns an array shaped (17, 3) with (y, x, confidence). Values are in
        normalized coordinates [0, 1]. If the model is unavailable, returns zeros.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return np.zeros((17, 3), dtype=np.float32)
        if not self._ready:
            return np.zeros((17, 3), dtype=np.float32)

        image = cv2.resize(crop_rgb, (self._input_size, self._input_size))
        image = image.astype(np.float32)
        image = image[np.newaxis, ...]

        try:
            outputs = self._model.signatures["serving_default"](tf.constant(image))
            keypoints_with_scores = outputs["output_0"].numpy()  # (1,1,17,3)
            keypoints = keypoints_with_scores[0, 0, :, :]  # (17,3)
            return keypoints.astype(np.float32)
        except Exception:
            return np.zeros((17, 3), dtype=np.float32)


def load_models(
    yolo_weights_path: str = "yolov8n.pt",
    movenet_variant: str = "singlepose_lightning",
) -> Tuple[Optional["YOLO"], MoveNetWrapper]:
    """Load and initialize the YOLO and MoveNet models.

    Returns a tuple (yolo_model, movenet_model). If ultralytics is not installed,
    yolo_model will be None. The MoveNet wrapper will always be returned;
    if TF/TF-Hub are missing it will produce zeros for keypoints.
    """
    yolo_model = None
    if YOLO is not None:
        try:
            # Resolve weights path robustly: check provided path, CWD, and script dir
            candidates = []
            if os.path.isabs(yolo_weights_path):
                candidates.append(yolo_weights_path)
            else:
                candidates.append(os.path.abspath(yolo_weights_path))
                candidates.append(os.path.join(os.getcwd(), yolo_weights_path))
                this_dir = os.path.dirname(__file__)
                candidates.append(os.path.join(this_dir, yolo_weights_path))
                # If default name used, also try script-dir root explicitly
                if yolo_weights_path != "yolov8n.pt":
                    candidates.append(os.path.join(this_dir, "yolov8n.pt"))

            resolved = None
            for cand in candidates:
                if isinstance(cand, str) and os.path.exists(cand):
                    resolved = cand
                    break

            yolo_model = YOLO(resolved or yolo_weights_path)
        except Exception:
            yolo_model = None

    movenet_model = MoveNetWrapper(model_name=movenet_variant)
    return yolo_model, movenet_model


def _compute_intersection_with_bounds(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return the two points where the infinite line through p1-p2 intersects the
    image rectangle bounds. If the line is parallel to the rectangle edges and
    does not intersect twice, returns None.
    """
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and y1 == y2:
        return None

    # Line parametrization: P = p1 + t*(p2-p1)
    dx = x2 - x1
    dy = y2 - y1

    intersections: List[Tuple[float, float]] = []

    # Check intersection with x = 0 and x = W-1
    for x in [0.0, float(frame_width - 1)]:
        if dx != 0:
            t = (x - x1) / dx
            y = y1 + t * dy
            if 0.0 <= y <= float(frame_height - 1):
                intersections.append((x, y))

    # Check intersection with y = 0 and y = H-1
    for y in [0.0, float(frame_height - 1)]:
        if dy != 0:
            t = (y - y1) / dy
            x = x1 + t * dx
            if 0.0 <= x <= float(frame_width - 1):
                intersections.append((x, y))

    # Deduplicate close points and keep up to two distinct intersections
    deduped: List[Tuple[float, float]] = []
    for pt in intersections:
        if all(np.hypot(pt[0] - q[0], pt[1] - q[1]) > 1e-6 for q in deduped):
            deduped.append(pt)

    if len(deduped) < 2:
        return None

    # If more than 2 due to corner overlaps, pick the two farthest apart
    if len(deduped) > 2:
        max_d = -1.0
        best_pair = (deduped[0], deduped[1])
        for i in range(len(deduped)):
            for j in range(i + 1, len(deduped)):
                d = (deduped[i][0] - deduped[j][0]) ** 2 + (deduped[i][1] - deduped[j][1]) ** 2
                if d > max_d:
                    max_d = d
                    best_pair = (deduped[i], deduped[j])
        return best_pair

    return deduped[0], deduped[1]


def extend_line_to_frame_bounds(
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Extend a line segment to intersect the frame boundaries.

    This handles perspective geometry such that a line representing a doubles
    sideline can be extended up to the top and bottom of the frame, producing a
    valid trapezoid-like playable region when paired with another sideline.

    Returns two integer points (x,y) on the frame boundary or None if extension
    is not possible.
    """
    p1, p2 = line
    result = _compute_intersection_with_bounds(p1, p2, frame_width, frame_height)
    if result is None:
        return None
    a, b = result
    return (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))


def _angle_from_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]
    angle_rad = np.arctan2(vx, vy)  # angle relative to vertical axis
    return float(np.degrees(abs(angle_rad)))


def detect_playable_area(frame: np.ndarray, margin_pixels: int = 50) -> np.ndarray:
    """Detect the playable court polygon based on two doubles sidelines.

    Steps:
    - Preprocess: grayscale, blur, Canny edges
    - Detect lines with Hough Transform
    - Filter near-vertical lines; select two that are left/right extremes
    - Extend these lines to frame bounds
    - Build a quadrilateral polygon and apply a horizontal margin

    Returns array of shape (4, 2) as integer vertices in clockwise order:
    [top_left, top_right, bottom_right, bottom_left].
    """
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(0.25 * w),
        maxLineGap=20,
    )

    if lines_p is None or len(lines_p) == 0:
        # Fallback to whole frame with margin clipped
        poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
        if margin_pixels > 0:
            poly[:, 0] = np.clip(poly[:, 0] + np.array([margin_pixels, -margin_pixels, -margin_pixels, margin_pixels]), 0, w - 1)
        return poly

    # Collect near-vertical lines
    vertical_candidates: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for l in lines_p:
        x1, y1, x2, y2 = l[0]
        angle_deg = _angle_from_vertical((x1, y1), (x2, y2))
        if angle_deg < 20.0:  # near vertical
            vertical_candidates.append(((x1, y1), (x2, y2)))

    if len(vertical_candidates) < 2:
        # Fallback to frame bounds
        poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
        if margin_pixels > 0:
            poly[:, 0] = np.clip(poly[:, 0] + np.array([margin_pixels, -margin_pixels, -margin_pixels, margin_pixels]), 0, w - 1)
        return poly

    # Score lines by x at mid-height to find left/right extremes
    mid_y = h * 0.5
    scored: List[Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]] = []
    for (x1, y1), (x2, y2) in vertical_candidates:
        if y2 == y1:
            continue
        t = (mid_y - y1) / (y2 - y1)
        x_at_mid = x1 + t * (x2 - x1)
        scored.append((float(x_at_mid), ((x1, y1), (x2, y2))))

    if len(scored) < 2:
        poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
        if margin_pixels > 0:
            poly[:, 0] = np.clip(poly[:, 0] + np.array([margin_pixels, -margin_pixels, -margin_pixels, margin_pixels]), 0, w - 1)
        return poly

    scored.sort(key=lambda t2: t2[0])
    left_line = scored[0][1]
    right_line = scored[-1][1]

    left_ext = extend_line_to_frame_bounds(left_line, w, h)
    right_ext = extend_line_to_frame_bounds(right_line, w, h)

    if left_ext is None or right_ext is None:
        poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
        if margin_pixels > 0:
            poly[:, 0] = np.clip(poly[:, 0] + np.array([margin_pixels, -margin_pixels, -margin_pixels, margin_pixels]), 0, w - 1)
        return poly

    # Determine which point is top/bottom for each extended line
    def sort_by_y(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (a, b) if a[1] <= b[1] else (b, a)

    lt, lb = sort_by_y(*left_ext)
    rt, rb = sort_by_y(*right_ext)

    poly = np.array([lt, rt, rb, lb], dtype=np.int32)

    # Apply horizontal margin: move left side margin_pixels left, right side margin_pixels right
    if margin_pixels != 0:
        poly[0, 0] = max(0, poly[0, 0] - margin_pixels)  # top-left x
        poly[3, 0] = max(0, poly[3, 0] - margin_pixels)  # bottom-left x
        poly[1, 0] = min(w - 1, poly[1, 0] + margin_pixels)  # top-right x
        poly[2, 0] = min(w - 1, poly[2, 0] + margin_pixels)  # bottom-right x

    return poly


def _point_in_polygon(point: Tuple[float, float], polygon_xy: np.ndarray) -> bool:
    contour = polygon_xy.reshape((-1, 1, 2)).astype(np.float32)
    res = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return res >= 0.0


def _read_annotations_csv(csv_path: str) -> List[Tuple[int, int, int]]:
    """Read annotations CSV as tuples of (start_frame, end_frame, label_id).

    Expected header with columns: start_frame,end_frame,label
    """
    import csv

    spans: List[Tuple[int, int, int]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = int(row["start_frame"])  # type: ignore[index]
                e = int(row["end_frame"])  # type: ignore[index]
                label_str = str(row["label"]).strip().upper()  # type: ignore[index]
                label_id = LABEL_TO_ID.get(label_str, 0)
                if e >= s:
                    spans.append((s, e, label_id))
            except Exception:
                continue
    return spans


def _create_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    num_frames = features.shape[0]
    for start in range(0, max(0, num_frames - seq_len + 1), stride):
        end = start + seq_len
        if end > num_frames:
            break
        xs.append(features[start:end])
        # Majority label within the window
        window = labels[start:end]
        vals, counts = np.unique(window, return_counts=True)
        ys.append(int(vals[np.argmax(counts)]))
    if len(xs) == 0:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def _compute_feature_vector_for_players(
    frame_bgr: np.ndarray,
    players_xywh: List[Tuple[int, int, int, int]],
    movenet: MoveNetWrapper,
    prev_centers: List[Optional[Tuple[float, float]]],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, List[Optional[Tuple[float, float]]]]:
    """Build a per-frame feature vector for up to two players.

    Features per player (slot order: left x then right x by center-x):
    - center_x_norm, center_y_norm
    - vel_x, vel_y (center delta vs previous frame; 0 if unknown)
    - 17 keypoints: (x_norm, y_norm) flattened (confidence ignored in feature)
    """
    h, w = image_size[1], image_size[0]

    # Sort players by center x to create consistent left/right ordering
    centers_and_boxes: List[Tuple[float, float, Tuple[int, int, int, int]]] = []
    for (x, y, bw, bh) in players_xywh:
        cx = x + bw / 2.0
        cy = y + bh
        centers_and_boxes.append((cx, cy, (x, y, bw, bh)))
    centers_and_boxes.sort(key=lambda t: t[0])

    feature_per_player: List[np.ndarray] = []
    new_prev_centers: List[Optional[Tuple[float, float]]] = [None, None]

    for slot in range(2):
        if slot < len(centers_and_boxes):
            cx, cy, (x, y, bw, bh) = centers_and_boxes[slot]
            cx_n = np.clip(cx / float(w), 0.0, 1.0)
            cy_n = np.clip(cy / float(h), 0.0, 1.0)
            prev_c = prev_centers[slot] if slot < len(prev_centers) else None
            if prev_c is None:
                vx, vy = 0.0, 0.0
            else:
                vx = cx_n - prev_c[0]
                vy = cy_n - prev_c[1]

            # Crop person and run MoveNet
            x0 = int(max(0, x))
            y0 = int(max(0, y))
            x1 = int(min(w - 1, x + bw))
            y1 = int(min(h - 1, y + bh))
            person_rgb = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            kps = movenet.infer_keypoints(person_rgb)  # (17,3) in (y,x,conf)

            # Map normalized keypoints back into frame normalized coordinate space
            # bbox top-left + keypoint*(bbox size) then normalize by frame size
            kps_xy = []
            if kps.shape == (17, 3):
                for (ky, kx, _conf) in kps.tolist():
                    kx_abs = x0 + kx * max(1, (x1 - x0))
                    ky_abs = y0 + ky * max(1, (y1 - y0))
                    kx_n = np.clip(kx_abs / float(w), 0.0, 1.0)
                    ky_n = np.clip(ky_abs / float(h), 0.0, 1.0)
                    kps_xy.append(kx_n)
                    kps_xy.append(ky_n)
            else:
                kps_xy = [0.0] * (17 * 2)

            feat = np.array([cx_n, cy_n, vx, vy] + kps_xy, dtype=np.float32)
            feature_per_player.append(feat)
            new_prev_centers[slot] = (cx_n, cy_n)
        else:
            # No player for this slot → zeros
            feature_per_player.append(np.zeros(4 + 17 * 2, dtype=np.float32))
            new_prev_centers[slot] = None

    # Concatenate left and right features
    return np.concatenate(feature_per_player, axis=0), new_prev_centers


def process_video_for_training(
    video_path: str,
    csv_path: str,
    output_npz_path: str,
    sequence_length: int = 60,
    sequence_stride: int = 15,
) -> None:
    """Run the end-to-end preprocessing for one video.

    - Detect playable area on first frame
    - YOLO detect persons per frame, filter by polygon, keep up to 2 players
    - MoveNet pose for players; compute per-frame features including velocity
    - Create overlapping sequences; label from CSV spans by majority vote
    - Save X,y in a compressed .npz
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotations CSV not found: {csv_path}")

    yolo_model, movenet_model = load_models()
    if yolo_model is None:
        raise RuntimeError(
            "YOLO model is unavailable. Please install 'ultralytics' and ensure yolov8n.pt exists."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Read first frame for playable area detection
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Could not read first frame for playable area detection.")

    playable_poly = detect_playable_area(first_frame, margin_pixels=50)  # (4,2)
    poly_contour = playable_poly.reshape((-1, 1, 2)).astype(np.float32)

    # Prepare to iterate frames
    features_per_frame: List[np.ndarray] = []
    # Initialize previous centers for 2 slots (left/right)
    prev_centers: List[Optional[Tuple[float, float]]] = [None, None]

    # Process the first frame as well: we already read it
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]

        # YOLO inference
        try:
            results = yolo_model.predict(source=frame, verbose=False)[0]  # type: ignore[union-attr]
        except Exception as e:
            cap.release()
            raise RuntimeError(f"YOLO inference failed at frame {frame_index}: {e}")

        players: List[Tuple[int, int, int, int, float]] = []  # x,y,w,h,conf
        for box in getattr(results, "boxes", []):  # type: ignore[attr-defined]
            # Ultralytics boxes: xyxy, conf, cls
            try:
                cls_id = int(box.cls.item())  # type: ignore[union-attr]
                if cls_id != 0:  # 0 = person
                    continue
                conf = float(box.conf.item())  # type: ignore[union-attr]
                xyxy = box.xyxy.cpu().numpy().reshape(-1)
                x0, y0, x1, y1 = [int(v) for v in xyxy]
                bw = max(1, x1 - x0)
                bh = max(1, y1 - y0)
                # bottom-center for polygon filter
                bottom_center = (x0 + bw / 2.0, y0 + bh)
                inside = cv2.pointPolygonTest(poly_contour, bottom_center, False) >= 0.0
                if inside:
                    players.append((x0, y0, bw, bh, conf))
            except Exception:
                continue

        # Keep up to 2 by confidence
        players.sort(key=lambda t: t[4], reverse=True)
        players_xywh = [(x, y, bw, bh) for (x, y, bw, bh, _c) in players[:2]]

        feat_vec, prev_centers = _compute_feature_vector_for_players(
            frame_bgr=frame,
            players_xywh=players_xywh,
            movenet=movenet_model,
            prev_centers=prev_centers,
            image_size=(w, h),
        )
        features_per_frame.append(feat_vec)

        frame_index += 1

    cap.release()

    if len(features_per_frame) == 0:
        raise RuntimeError("No frames processed; cannot create training data.")

    features = np.stack(features_per_frame).astype(np.float32)

    # Labels per frame from annotation CSV
    spans = _read_annotations_csv(csv_path)
    frame_labels = np.zeros((features.shape[0],), dtype=np.int64)
    for s, e, lid in spans:
        if s >= len(frame_labels):
            continue
        e_clamped = min(e, len(frame_labels) - 1)
        frame_labels[s : e_clamped + 1] = lid

    X, y = _create_sequences(features, frame_labels, sequence_length, sequence_stride)

    # Save compressed NPZ
    os.makedirs(os.path.dirname(output_npz_path) or ".", exist_ok=True)
    np.savez_compressed(output_npz_path, X=X, y=y, fps=float(fps), total_frames=int(total_frames))


__all__ = [
    "load_models",
    "extend_line_to_frame_bounds",
    "detect_playable_area",
    "process_video_for_training",
]


