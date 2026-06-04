"""Generate the court-detection regression fixtures.

For each hand-annotated video, extract the clean frame the detector would run on
(the image BEFORE the court mask) and the resulting golden out-mask, and freeze
both to tests/fixtures/court/. The frozen PNG frame is a deterministic input, so
the regression test can re-run detection on it without YOLO/video/RANSAC and
compare against the golden mask within a tolerance.

Run with a venv that has cv2/ultralytics/av/torch, with the source locations in env:
  RALLYCLIP_COURT_VIDEO_DIR  -> the hand-annotated source videos
  RALLYCLIP_YOLO_WEIGHTS     -> the YOLO pose weights
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.ERROR)
cv2.setRNGSeed(0)  # make regeneration as stable as possible

# Source videos / weights live outside the repo and are machine-specific, so they come
# from env (the same vars the e2e flight uses). The repo root is derived from this file.
VIDEO_DIR = os.environ.get("RALLYCLIP_COURT_VIDEO_DIR")
YOLO_WEIGHTS = os.environ.get("RALLYCLIP_YOLO_WEIGHTS")
REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "court"
sys.path.insert(0, str(REPO / "src"))

from preprocessing.court_detector_impl import CourtDetector  # noqa: E402

SAMPLE_TIMES = [60, 90, 45, 120, 30, 150]


def slug(name: str) -> str:
    base = re.split(r"_utr|_dur", name)[0]
    base = base.encode("ascii", "ignore").decode()
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()
    return base or "court"


def main() -> int:
    if not VIDEO_DIR or not YOLO_WEIGHTS:
        raise SystemExit(
            "Set RALLYCLIP_COURT_VIDEO_DIR and RALLYCLIP_YOLO_WEIGHTS (court source videos "
            "and YOLO pose weights) before regenerating fixtures."
        )
    (FIX / "frames").mkdir(parents=True, exist_ok=True)
    (FIX / "masks").mkdir(parents=True, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    detector = CourtDetector(yolo_model_path=YOLO_WEIGHTS, conf=0.25)

    manifest, used = [], set()
    for vp in videos:
        filename = Path(vp).name      # full name (with extension): manifest + e2e video lookup
        sid = slug(Path(vp).stem)     # slug from the stem so no "_mp4" suffix leaks in
        while sid in used:
            sid += "_x"
        used.add(sid)

        mask = frame = None
        meta, t_ok = {}, None
        for t in SAMPLE_TIMES:
            try:
                mask, frame, meta = detector.process_video(vp, target_time=t)
            except Exception as e:  # noqa: BLE001
                meta, mask, frame = {"error": f"exc:{e}"}, None, None
            if mask is not None and np.any(mask):
                t_ok = t
                break

        if mask is None or not np.any(mask):
            print(f"  SKIP (no detection) {sid}  err={meta.get('error')}")
            manifest.append({"id": sid, "video": filename, "detected": False, "error": meta.get("error")})
            continue

        cv2.imwrite(str(FIX / "frames" / f"{sid}.png"), frame)
        cv2.imwrite(str(FIX / "masks" / f"{sid}.png"), mask)
        manifest.append({
            "id": sid, "video": filename, "detected": True, "timestamp_s": t_ok,
            "frame_shape": list(frame.shape), "baseline_width": int(meta.get("baseline_width", 0)),
        })
        print(f"  OK  {sid:34s} t={t_ok}s")

    (FIX / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ok = sum(1 for m in manifest if m["detected"])
    print(f"\nFixtures: {ok}/{len(manifest)} detected -> {FIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
