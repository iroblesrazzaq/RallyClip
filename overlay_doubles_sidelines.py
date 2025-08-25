import os
import argparse
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _compute_intersection_with_bounds(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and y1 == y2:
        return None

    dx = x2 - x1
    dy = y2 - y1
    intersections: List[Tuple[float, float]] = []

    # x = 0 and x = W-1
    for x in [0.0, float(frame_width - 1)]:
        if dx != 0:
            t = (x - x1) / dx
            y = y1 + t * dy
            if 0.0 <= y <= float(frame_height - 1):
                intersections.append((x, y))

    # y = 0 and y = H-1
    for y in [0.0, float(frame_height - 1)]:
        if dy != 0:
            t = (y - y1) / dy
            x = x1 + t * dx
            if 0.0 <= x <= float(frame_width - 1):
                intersections.append((x, y))

    deduped: List[Tuple[float, float]] = []
    for pt in intersections:
        if all(np.hypot(pt[0] - q[0], pt[1] - q[1]) > 1e-6 for q in deduped):
            deduped.append(pt)

    if len(deduped) < 2:
        return None
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


def _extend_line_to_bounds(
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    res = _compute_intersection_with_bounds(line[0], line[1], frame_width, frame_height)
    if res is None:
        return None
    a, b = res
    return (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))


def _angle_from_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]
    angle_rad = np.arctan2(vx, vy)
    return float(np.degrees(abs(angle_rad)))


def detect_doubles_sidelines(frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return two extended near-vertical lines (left, right) as endpoints.

    If detection fails, returns None.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(gray, 50, 150)

    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(0.25 * w),
        maxLineGap=20,
    )

    if lines_p is None or len(lines_p) == 0:
        return None

    candidates: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for l in lines_p:
        x1, y1, x2, y2 = l[0]
        if _angle_from_vertical((x1, y1), (x2, y2)) < 20.0:
            candidates.append(((x1, y1), (x2, y2)))

    if len(candidates) < 2:
        return None

    mid_y = h * 0.5
    scored: List[Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]] = []
    for (x1, y1), (x2, y2) in candidates:
        if y2 == y1:
            continue
        t = (mid_y - y1) / (y2 - y1)
        x_at_mid = x1 + t * (x2 - x1)
        scored.append((float(x_at_mid), ((x1, y1), (x2, y2))))

    if len(scored) < 2:
        return None

    scored.sort(key=lambda t2: t2[0])
    left_line = scored[0][1]
    right_line = scored[-1][1]

    left_ext = _extend_line_to_bounds(left_line, w, h)
    right_ext = _extend_line_to_bounds(right_line, w, h)
    if left_ext is None or right_ext is None:
        return None

    return left_ext, right_ext


def overlay_sidelines(
    video_path: str,
    output_path: str,
    seconds: int = 60,
    thickness: int = 16,
) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(seconds * fps)

    ret, first = cap.read()
    if not ret or first is None:
        cap.release()
        raise RuntimeError("Could not read first frame for detection")

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Detect lines once from first frame
    res = detect_doubles_sidelines(first)
    if res is None:
        print("Warning: could not detect doubles sidelines. Writing original frames.")
        left_ext = right_ext = None
    else:
        left_ext, right_ext = res

    # Process frames from beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while True:
        if count >= max_frames:
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if left_ext is not None and right_ext is not None:
            cv2.line(frame, left_ext[0], left_ext[1], color=(0, 0, 0), thickness=thickness)
            cv2.line(frame, right_ext[0], right_ext[1], color=(0, 0, 0), thickness=thickness)

        writer.write(frame)
        count += 1

    cap.release()
    writer.release()
    print(f"Saved overlay video to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay black bars on detected doubles sidelines.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default=os.path.abspath("sanity_check_clips/doubles_sidelines_overlay.mp4"), help="Path to output video")
    parser.add_argument("--seconds", type=int, default=60, help="How many seconds to process")
    parser.add_argument("--thickness", type=int, default=16, help="Line thickness in pixels")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    overlay_sidelines(args.video, args.out, seconds=args.seconds, thickness=args.thickness)


if __name__ == "__main__":
    main()


