#!/usr/bin/env python3
"""Probe: how much does each decode path contribute to peak RSS? (read-only, no pipeline changes)

Runs the real release pipeline under the RSS sampler with optional runtime stubs:
  --stub-yolo  : weightless fake YOLO (isolates decode/data from the model weights)
  --stub-court : replace compute_court_mask with a no-op zeros mask -> removes the cv2
                 VideoCapture court-detection decode (the PyAV pose decode still runs)

Compare peak_rss_delta across combinations to learn (a) the cv2 court decode's share of peak
and (b) whether peak RSS lands during court detection or the pose pass. Nothing in src/ is
modified; stubs are applied only inside this process.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PERF = ROOT / "scripts" / "perf"
for p in (str(SRC), str(PERF)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bench_pipeline as bp  # noqa: E402
import bench_real as br  # noqa: E402


class _FakeYOLO(bp._FakeModel):
    def __init__(self, *a, **k) -> None:
        super().__init__()

    def __call__(self, *a, **k):
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--stub-yolo", action="store_true")
    ap.add_argument("--stub-court", action="store_true")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    os.environ.setdefault("RALLYCLIP_NO_TQDM", "1")
    os.chdir(ROOT)

    cli = importlib.import_module("cli.main")
    real_loader = cli._load_config_dict
    cli._load_config_dict = lambda path: ({} if path == "__none__" else real_loader(path))

    if args.stub_yolo:
        import extraction.pose_extractor as pe
        import preprocessing.court_detector_impl as cd
        pe.YOLO = _FakeYOLO
        cd.YOLO = _FakeYOLO

    if args.stub_court:
        from preprocessing.data_preprocessor import DataPreprocessor
        from runtime.video_validation import probe_video

        def _stub_mask(self, video_path):
            try:
                info = probe_video(video_path)
                h, w = int(info.height), int(info.width)
            except Exception:
                h, w = self.screen_height, self.screen_width
            return np.zeros((h, w), dtype=np.uint8), {"source": "stub", "detected": False, "timestamp_s": None}

        DataPreprocessor.compute_court_mask = _stub_mask  # type: ignore[method-assign]

    out_dir = Path(tempfile.mkdtemp(prefix="rcprobe_"))
    duration = args.duration if args.duration > 0 else None
    cfg = cli.build_run_config(br._make_args(video, out_dir, args.start, duration, args.device))

    wall0 = time.perf_counter()
    with bp.RSSSampler() as rss:
        rc = cli.run_pipeline(cfg)
    elapsed = time.perf_counter() - wall0

    tag = f"yolo={'stub' if args.stub_yolo else 'real'} court={'stub' if args.stub_court else 'real'}"
    print(f"[probe] {tag}  dur={args.duration}s  rc={rc}  peak_rss={round(rss.peak_delta_mb, 2)}MB  elapsed={round(elapsed, 2)}s")

    shutil.rmtree(out_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
