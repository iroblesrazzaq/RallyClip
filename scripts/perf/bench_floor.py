#!/usr/bin/env python3
"""Measure peak RSS of the real release pipeline WITHOUT loading YOLO weights.

Stubs the ultralytics ``YOLO`` class (no ``.pt`` load, no network allocation) in BOTH the pose
extractor and the court detector, while keeping everything else real: the torch / onnxruntime /
ultralytics imports (already resident before the sampler starts), real PyAV 1080p decode, the
streaming pipeline, and the LSTM/ONNX segment inference. Correctness is irrelevant here (fake
detections) -- this isolates how much of the streamed-pipeline RSS floor is the YOLO model
weights vs everything else.

Compare its ``peak_rss_delta`` against ``bench_real.py`` at the SAME ``--duration`` (with real
YOLO). The difference is the YOLO model's contribution to peak RSS.

Usage:
  python scripts/perf/bench_floor.py --video raw_video/testing_app/1_*.mp4 --duration 120
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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PERF = ROOT / "scripts" / "perf"
for p in (str(SRC), str(PERF)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bench_pipeline as bp  # noqa: E402  (RSSSampler, _FakeModel)
import bench_real as br  # noqa: E402  (_make_args + config-loader sentinel)


class _FakeYOLO(bp._FakeModel):
    """Stand-in for ultralytics.YOLO that loads NO weights into memory.

    Inherits _FakeModel.predict (one synthetic 2-player result per frame) and .to(); adds a
    weightless constructor and a __call__ so the court detector's call paths don't load a model
    either (any unexpected call simply yields nothing and court detection falls back to default).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, *args, **kwargs):
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--duration", type=int, default=120, help="seconds of source video (0 = full)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
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

    # Stub YOLO at both instantiation sites so no .pt weights are ever loaded.
    import extraction.pose_extractor as pe
    import preprocessing.court_detector_impl as cd

    pe.YOLO = _FakeYOLO
    cd.YOLO = _FakeYOLO

    out_dir = Path(tempfile.mkdtemp(prefix="rcfloor_"))
    duration = args.duration if args.duration > 0 else None
    cfg = cli.build_run_config(br._make_args(video, out_dir, args.start, duration, args.device))

    wall0 = time.perf_counter()
    with bp.RSSSampler() as rss:
        rc = cli.run_pipeline(cfg)
    elapsed = time.perf_counter() - wall0

    print(f"[floor] NO-YOLO  device={args.device}  dur={args.duration}s  rc={rc}")
    print(f"  peak_rss_delta : {round(rss.peak_delta_mb, 2)} MB")
    print(f"  elapsed        : {round(elapsed, 2)} s")

    shutil.rmtree(out_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
