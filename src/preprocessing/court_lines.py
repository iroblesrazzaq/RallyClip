"""Court geometry from the line-map court model (ONNX, torch-free).

The model predicts one probability map per court line (near/far baseline,
left/right sideline, net). Maps are averaged over a few frames spread across
the video, a line is fit to each, and the six court points are the line
intersections (a point may fall outside the frame). A net point that disagrees
with the four corners is replaced by the corner-implied one. The result is the
net line plus a pixels -> court-meters homography, which the net-line player
slots need, and the court "out" mask that filters detections off the court.

Same algorithm as training_data/court_keypoints (lines.decode + court_onnx),
which produced the geometry the v0.6 model was trained on.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

INPUT_W, INPUT_H = 512, 288
POINT_NAMES = ("near_left", "near_right", "far_left", "far_right", "net_left", "net_right")
# Which two lines (near_baseline, far_baseline, left, right, net) meet at each point.
POINT_LINES = ((0, 2), (0, 3), (1, 2), (1, 3), (4, 2), (4, 3))
HALF_W, HALF_L = 10.97 / 2, 23.77 / 2
COURT_M = np.array(
    [[-HALF_W, -HALF_L], [HALF_W, -HALF_L], [-HALF_W, HALF_L], [HALF_W, HALF_L], [-HALF_W, 0.0], [HALF_W, 0.0]],
    np.float64,
)
NET_TOLERANCE_PX = 60.0
SAMPLE_FRACTIONS = tuple(np.linspace(0.1, 0.9, 7))


# --- line fitting (pure numpy) -------------------------------------------------
def _tls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted total least squares line (a, b, c) with a^2 + b^2 = 1."""
    ws = w.sum()
    cx = (w * x).sum() / ws
    cy = (w * y).sum() / ws
    dx = x - cx
    dy = y - cy
    cov = np.array([[(w * dx * dx).sum(), (w * dx * dy).sum()], [(w * dx * dy).sum(), (w * dy * dy).sum()]])
    _vals, vecs = np.linalg.eigh(cov)
    a, b = vecs[:, 0]
    return np.array([a, b, -(a * cx + b * cy)])


def fit_line(prob: np.ndarray, rng: np.random.Generator, thresh: float = 0.25, inlier: float = 1.5):
    """RANSAC then TLS on one probability map. Returns (line in map pixels or None, confidence)."""
    peak = float(prob.max())
    if peak < thresh:
        return None, peak
    ys, xs = np.nonzero(prob >= max(thresh, 0.3 * peak))
    if xs.size < 6:
        return None, peak
    x = xs.astype(np.float64) + 0.5
    y = ys.astype(np.float64) + 0.5
    w = prob[ys, xs].astype(np.float64)
    best_score = -1.0
    best_in = None
    pick = rng.choice(xs.size, size=(200, 2), p=w / w.sum())
    for i, j in pick:
        if i == j:
            continue
        a = y[j] - y[i]
        b = x[i] - x[j]
        norm = np.hypot(a, b)
        if norm < 2.0:
            continue
        a /= norm
        b /= norm
        c = -(a * x[i] + b * y[i])
        inl = np.abs(a * x + b * y + c) < inlier
        score = float(w[inl].sum())
        if score > best_score:
            best_score = score
            best_in = inl
    if best_in is None or best_in.sum() < 6:
        return None, peak
    line = _tls(x[best_in], y[best_in], w[best_in])
    inl = np.abs(line[0] * x + line[1] * y + line[2]) < inlier
    if inl.sum() >= 6:
        line = _tls(x[inl], y[inl], w[inl])
    return line, float(w[inl].mean())


def decode(probs: np.ndarray, seed: int = 0):
    """probs: [5, H, W] sigmoid maps -> (line_ok [5], points [6, 2] normalized, point_ok [6])."""
    rng = np.random.default_rng(seed)
    _k, map_h, map_w = probs.shape
    lines_px = np.zeros((5, 3))
    line_ok = np.zeros(5, bool)
    for i in range(5):
        line, _conf = fit_line(probs[i], rng)
        if line is not None:
            lines_px[i] = line
            line_ok[i] = True
    points = np.full((6, 2), np.nan)
    point_ok = np.zeros(6, bool)
    for k, (i, j) in enumerate(POINT_LINES):
        if not (line_ok[i] and line_ok[j]):
            continue
        p = np.cross(lines_px[i], lines_px[j])
        if abs(p[2]) < 1e-9:
            continue
        points[k] = p[0] / p[2] / map_w, p[1] / p[2] / map_h
        point_ok[k] = np.all(np.abs(points[k]) < 20)
    return line_ok, points, point_ok


# --- model + geometry ----------------------------------------------------------
class CourtLineModel:
    """ONNX graph: rgb [N, 3, 288, 512] float 0..255 -> sigmoid line maps [N, 5, 72, 128]."""

    def __init__(self, onnx_path: str, providers: Optional[list[str]] = None) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])

    def line_probs(self, frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Mean line maps [5, 72, 128] over the given frames."""
        batch = np.stack([
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
            for f in frames_bgr
        ]).transpose(0, 3, 1, 2).astype(np.float32)
        return self.session.run(None, {"rgb": np.ascontiguousarray(batch)})[0].mean(0)

    def court(self, frames_bgr: Sequence[np.ndarray], out_size: Optional[tuple[int, int]] = None) -> dict:
        """Court geometry in pixels of out_size (width, height); default: the frames' own size."""
        size = out_size or (frames_bgr[0].shape[1], frames_bgr[0].shape[0])
        return geometry_from_decode(decode(self.line_probs(frames_bgr)), size, len(frames_bgr))


def geometry_from_decode(decoded, size: tuple[int, int], frames_used: int) -> dict:
    """Geometry dict in pixels of size (width, height) from decode() output."""
    w, h = size
    line_ok, points, point_ok = decoded
    point_ok = point_ok.copy()
    pts_px = points * np.array([w, h])
    net_source = ["model", "model"]
    if point_ok[:4].all():
        h_c, _ = cv2.findHomography(COURT_M[:4], pts_px[:4].astype(np.float64), 0)
        implied = cv2.perspectiveTransform(COURT_M[None, 4:6], h_c)[0]
        for k in (4, 5):
            if not point_ok[k] or np.linalg.norm(implied[k - 4] - pts_px[k]) > NET_TOLERANCE_PX:
                pts_px[k] = implied[k - 4]
                point_ok[k] = True
                net_source[k - 4] = "corners"
    out = {
        "width": int(w),
        "height": int(h),
        "frames_used": int(frames_used),
        "points_px": {n: (pts_px[i].round(1).tolist() if point_ok[i] else None) for i, n in enumerate(POINT_NAMES)},
        "lines_ok": line_ok.tolist(),
        "ok": bool(point_ok.all()),
        "net_source": net_source,
        "net_line_px": None,
        "homography_px_to_m": None,
    }
    if point_ok[4] and point_ok[5]:
        out["net_line_px"] = [pts_px[4].round(2).tolist(), pts_px[5].round(2).tolist()]
    if point_ok.sum() >= 4:
        hm, _ = cv2.findHomography(pts_px[point_ok].astype(np.float64), COURT_M[point_ok], 0)
        out["homography_px_to_m"] = hm.tolist() if hm is not None else None
    return out


# --- court "out" mask ------------------------------------------------------------
BASE_HORIZONTAL_SHIFT = 100.0  # outward sideline padding (px) for a near baseline as wide as the frame


def padded_sidelines(points_px: dict, width: int) -> Optional[list]:
    """Left and right doubles sidelines moved outward, as (slope, intercept) of y = slope * x + intercept,
    or (None, x) for a vertical line. None when a corner is missing.

    The classic detector's padding: BASE_HORIZONTAL_SHIFT times the visible
    near-baseline width over the frame width, applied to the intercept.
    """
    corners = [points_px.get(n) for n in ("near_left", "far_left", "near_right", "far_right")]
    if any(c is None for c in corners):
        return None
    near_left, far_left, near_right, far_right = corners
    visible = np.clip([near_left[0], near_right[0]], 0, width)
    shift = BASE_HORIZONTAL_SHIFT * abs(float(visible[1] - visible[0])) / width
    lines = []
    for (x1, y1), (x2, y2), side in ((near_left, far_left, -1), (near_right, far_right, 1)):
        if x2 == x1:
            lines.append((None, x1 + side * shift))
            continue
        slope = (y2 - y1) / (x2 - x1)
        lines.append((slope, y1 - slope * x1 - shift / np.sqrt(1 + slope**2)))
    return lines


def out_mask(points_px: dict, width: int, height: int) -> Optional[np.ndarray]:
    """uint8 [height, width] mask, 255 outside the padded doubles sidelines, else 0.

    Everything on the far side of either padded sideline from the court is out:
    above the line in the image, or beyond it for a vertical line. points_px are
    pixels of (width, height). None when a corner is missing or nothing is out.
    """
    lines = padded_sidelines(points_px, width)
    if lines is None:
        return None
    ys = np.arange(height)[:, None]
    xs = np.arange(width)[None, :]
    out = np.zeros((height, width), bool)
    for (slope, value), side in zip(lines, (-1, 1)):
        if slope is None:
            out |= (xs - value) * side > 0
        else:
            out |= ys < slope * xs + value
    if not out.any():
        return None
    return out.astype(np.uint8) * 255


class CourtDetection:
    """One court-model pass over a video: geometry at any size, and the out mask."""

    def __init__(self, decoded, frame_size: tuple[int, int], frames_used: int) -> None:
        self.decoded = decoded
        self.frame_size = frame_size  # (width, height) of the decoded frames
        self.frames_used = frames_used

    def geometry(self, out_size: Optional[tuple[int, int]] = None) -> dict:
        return geometry_from_decode(self.decoded, out_size or self.frame_size, self.frames_used)

    def out_mask(self) -> Optional[np.ndarray]:
        """Out mask in the frames' own pixels."""
        return out_mask(self.geometry()["points_px"], *self.frame_size)


def sample_frames(video_path: str, fractions: Sequence[float] = SAMPLE_FRACTIONS) -> list[np.ndarray]:
    """BGR frames at the given fractions of the video duration.

    Picks the frame `ffmpeg -ss <t:.2f> -i video -frames:v 1` returns (how the
    training geometry was sampled): -ss is relative to the container start
    time, and the first frame at or after that point is kept.
    """
    import av

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        start = (container.start_time or 0) / 1e6
        duration = None
        if container.duration:
            duration = container.duration / 1e6
        elif stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        if not duration:
            return frames
        for frac in fractions:
            target = start + round(float(frac) * duration, 2)
            container.seek(int(target * 1e6), backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.time is not None and frame.time + 1e-6 >= target:
                    frames.append(frame.to_ndarray(format="bgr24"))
                    break
    return frames


def detect_court(
    video_path: str,
    model_path: str,
    providers: Optional[list[str]] = None,
) -> Optional[CourtDetection]:
    """Run the court model once on frames sampled across the video; None on failure."""
    try:
        frames = sample_frames(video_path)
        if not frames:
            logger.warning("Court model: no frames decoded from %s", video_path)
            return None
        decoded = decode(CourtLineModel(model_path, providers).line_probs(frames))
    except Exception as exc:  # a court failure must not abort the run
        logger.warning("Court model failed on %s: %s", video_path, exc)
        return None
    return CourtDetection(decoded, (frames[0].shape[1], frames[0].shape[0]), len(frames))


def netline_geometry(court: Optional[CourtDetection], out_size: Optional[tuple[int, int]] = None) -> Optional[dict]:
    """Geometry dict (see geometry_from_decode) or None without a net line + homography."""
    if court is None:
        return None
    try:
        geometry = court.geometry(out_size)
    except Exception as exc:  # a court failure must not abort the run
        logger.warning("Court geometry failed: %s", exc)
        return None
    if not geometry.get("net_line_px") or not geometry.get("homography_px_to_m"):
        logger.warning("Court model found no net line/homography (lines_ok=%s)", geometry.get("lines_ok"))
        return None
    return geometry


def compute_court_geometry(
    video_path: str,
    model_path: str,
    out_size: Optional[tuple[int, int]] = None,
    providers: Optional[list[str]] = None,
) -> Optional[dict]:
    """Geometry dict or None when there is no usable net line + homography."""
    return netline_geometry(detect_court(video_path, model_path, providers), out_size)
