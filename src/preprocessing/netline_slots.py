"""Net-line player slots, shared by training preprocess and the runtime.

Near slot: the usual near-player pick among full-frame detections whose box
bottom is below the net line. Far slot: the far-crop detection that sits above
the net line and, in court meters, is closest to the center line.

numpy only (no h5py/torch) so the frozen app can import it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Court meters: origin at the net center, u to the camera's right, v away from
# the camera. Far-slot gate: doubles half-width plus room to run wide, and from
# just short of the net to the far baseline plus run-back.
FAR_MAX_ABS_U = 10.97 / 2 + 2.0
FAR_MIN_V = -1.0
FAR_MAX_V = 23.77 / 2 + 6.0
FAR_TIE_U = 1.0


@dataclass
class CourtGeometry:
    """Per-video court from the line-map court model (static camera)."""

    net_a: np.ndarray  # net-bottom line endpoints, pixels
    net_b: np.ndarray
    px_to_m: np.ndarray  # 3x3 homography, pixels -> court meters
    source_sha256: str = ""

    @classmethod
    def from_dict(cls, data: dict, source_sha256: str = "") -> "CourtGeometry":
        if not data.get("net_line_px") or not data.get("homography_px_to_m"):
            raise ValueError("Court geometry has no net line or homography")
        a, b = (np.asarray(p, dtype=np.float64) for p in data["net_line_px"])
        return cls(a, b, np.asarray(data["homography_px_to_m"], dtype=np.float64), source_sha256)

    @classmethod
    def load(cls, geometry_dir: Optional[str], stem: str) -> "CourtGeometry":
        if not geometry_dir:
            raise ValueError("slot_mode=netline needs preprocess.court_geometry_dir")
        path = Path(geometry_dir) / f"{stem}.json"
        raw = path.read_bytes()
        try:
            return cls.from_dict(json.loads(raw), hashlib.sha256(raw).hexdigest())
        except ValueError as exc:
            raise ValueError(f"{exc}: {path}") from None

    def net_y_at(self, x: float) -> float:
        dx = self.net_b[0] - self.net_a[0]
        t = 0.0 if abs(dx) < 1e-9 else (x - self.net_a[0]) / dx
        return float(self.net_a[1] + t * (self.net_b[1] - self.net_a[1]))

    def below_net(self, box: np.ndarray) -> bool:
        """Box bottom lower on screen than the net line at the box's center x."""
        return float(box[3]) > self.net_y_at((float(box[0]) + float(box[2])) / 2)

    def feet_to_court(self, box: np.ndarray) -> tuple[float, float]:
        x, y = (float(box[0]) + float(box[2])) / 2, float(box[3])
        u, v, w = self.px_to_m @ np.array([x, y, 1.0])
        return float(u / w), float(v / w)


def feet_on_court(box: np.ndarray, court_mask: Optional[np.ndarray]) -> bool:
    """Feet (bottom-centre) inside the playable court (mask: 0 == on court).

    Deliberately feet, not centroid: the court trapezoid narrows with height, so
    a far player near a sideline can have their feet cleanly on court while their
    centroid falls outside the narrower band above them.
    """
    if court_mask is None:
        return True
    fx = int(np.clip((box[0] + box[2]) / 2, 0, court_mask.shape[1] - 1))
    fy = int(np.clip(box[3], 0, court_mask.shape[0] - 1))
    return bool(court_mask[int(fy), int(fx)] == 0)


def below_net_mask(boxes: np.ndarray, geometry: CourtGeometry) -> np.ndarray:
    return np.array([geometry.below_net(b) for b in boxes], dtype=bool)


def pick_far(
    boxes: np.ndarray,
    court_mask: Optional[np.ndarray],
    geometry: CourtGeometry,
) -> Optional[int]:
    """Crop detection most likely to be the far singles player, or None.

    Feet on court and above the net line; then gated to the far half in court
    meters; closest to the center line wins, with candidates within FAR_TIE_U
    of the best broken by closeness to the net. Meters, not pixels: pixel
    offsets shrink with depth, so a spectator behind the fence would look
    closer to center than a player one meter off it.
    """
    scored = []
    for i in range(len(boxes)):
        if not feet_on_court(boxes[i], court_mask) or geometry.below_net(boxes[i]):
            continue
        u, v = geometry.feet_to_court(boxes[i])
        if abs(u) < FAR_MAX_ABS_U and FAR_MIN_V < v < FAR_MAX_V:
            scored.append((abs(u), v, i))
    if not scored:
        return None
    best_u = min(s[0] for s in scored)
    return min((s for s in scored if s[0] - best_u <= FAR_TIE_U), key=lambda s: s[1])[2]
