#!/usr/bin/env python3
"""Real-video benchmark for the RallyClip release pipeline (the honest memory test).

Runs the ACTUAL CLI `run_pipeline` end-to-end on a real video (real YOLO pose at the
manifest imgsz, real court detection, real decode) while measuring the same metrics as the
stub harness: peak RSS, intermediate serialization time/bytes (the NPZ round-trips the
refactor removes), total wall time, and a segment-CSV hash as the real correctness anchor.

This is slow (real model inference), so it is the *milestone / acceptance* gate, not the
per-iteration gate. Use the long 1080p clips in raw_video/testing_app/ for the memory test:

  python scripts/perf/bench_real.py --video raw_video/testing_app/1_*.mp4 --duration 120 --pretty
  python scripts/perf/bench_real.py --video raw_video/testing_app/1_*.mp4 --json real_1.json   # full

The model runs at target_fps=5, so a ~30-min clip is ~9k processed frames. Pass --duration
to bound a quick validation run (seconds of source video).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PERF = ROOT / "scripts" / "perf"
for p in (str(SRC), str(PERF)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bench_pipeline as bp  # RSSSampler, _install_numpy_io_probes, SER  # noqa: E402

ARTIFACT_DIR = ROOT / "models" / "rallyclip_v0.3.1"

# build_run_config reads these via getattr(args, key, None); anything unset must be None.
_ARG_KEYS = (
    "config", "video", "output_dir", "output_name", "csv_output_dir", "yolo_device",
    "write_csv", "segment_video", "artifact_dir", "model_path", "scaler_path",
    "manifest_path", "yolo_size", "fps", "seq_len", "overlap", "sigma", "low", "high",
    "min_dur_sec", "conf", "imgsz", "start_time", "duration",
)


def _make_args(video: Path, out_dir: Path, start: int, duration: int | None, device: str | None) -> Namespace:
    ns = {k: None for k in _ARG_KEYS}
    ns.update(
        config="__none__",  # sentinel: skip config.toml so the run is reproducible from manifest only
        video=str(video),
        output_dir=str(out_dir),
        csv_output_dir=str(out_dir),
        artifact_dir=str(ARTIFACT_DIR),
        write_csv=True,
        segment_video=False,  # skip the expensive re-encode; we only need segments + metrics
        yolo_device=device,
        start_time=start,
        duration=duration,
    )
    return Namespace(**ns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="path to a real .mp4")
    ap.add_argument("--duration", type=int, default=0, help="seconds of source video to process (0 = full)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None, help="force pose device")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    os.environ.setdefault("RALLYCLIP_NO_TQDM", "1")
    os.chdir(ROOT)  # models/, pose_data/, default config resolution all key off cwd

    # Patch cli.main._load_config_dict to honour the "__none__" sentinel (manifest-only run).
    # Use importlib: cli/__init__ exports the `main` function, which shadows the submodule on
    # attribute access (so `import cli.main as cli` would bind the function, not the module).
    import importlib

    cli = importlib.import_module("cli.main")

    real_loader = cli._load_config_dict
    cli._load_config_dict = lambda path: ({} if path == "__none__" else real_loader(path))

    bp._install_numpy_io_probes()
    bp.SER.update({"write_s": 0.0, "write_mb": 0.0, "read_s": 0.0, "read_mb": 0.0, "writes": 0, "reads": 0})

    pose_data_existed = (ROOT / "pose_data").exists()
    out_dir = Path(tempfile.mkdtemp(prefix="rcreal_"))
    duration = args.duration if args.duration > 0 else None

    cfg = cli.build_run_config(_make_args(video, out_dir, args.start, duration, args.device))

    wall0 = time.perf_counter()
    with bp.RSSSampler() as rss:
        rc = cli.run_pipeline(cfg)
    elapsed = time.perf_counter() - wall0

    csv_files = list(out_dir.glob("*_segments.csv"))
    csv_sha = "missing"
    n_seg = -1
    if csv_files:
        raw = csv_files[0].read_bytes()
        csv_sha = hashlib.sha256(raw).hexdigest()[:16]
        n_seg = max(0, raw.decode("utf-8", "replace").strip().count("\n"))  # minus header below
        n_seg = max(0, n_seg)

    metrics = {
        "mode": "real",
        "video": video.name,
        "duration_s": args.duration or "full",
        "device": cfg.yolo_device or "auto",
        "return_code": rc,
        "serialization": {
            "write_s": round(bp.SER["write_s"], 4),
            "write_mb": round(bp.SER["write_mb"], 3),
            "read_s": round(bp.SER["read_s"], 4),
            "read_mb": round(bp.SER["read_mb"], 3),
            "total_s": round(bp.SER["write_s"] + bp.SER["read_s"], 4),
            "total_mb": round(bp.SER["write_mb"] + bp.SER["read_mb"], 3),
            "writes": bp.SER["writes"],
            "reads": bp.SER["reads"],
        },
        "peak_rss_delta_mb": round(rss.peak_delta_mb, 2),
        "elapsed_total_s": round(elapsed, 2),
        "correctness": {"segments_csv_sha256": csv_sha, "csv_rows_incl_header": n_seg},
    }

    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.pretty or not args.json:
        s = metrics
        print(f"[real] {s['video']}  dur={s['duration_s']}  device={s['device']}  rc={s['return_code']}")
        print(f"  elapsed         : {s['elapsed_total_s']}s")
        print(f"  serialization   : total {s['serialization']['total_s']}s / {s['serialization']['total_mb']}MB "
              f"({s['serialization']['writes']}w/{s['serialization']['reads']}r)")
        print(f"  peak_rss_delta  : {s['peak_rss_delta_mb']} MB")
        print(f"  segments csv    : sha={s['correctness']['segments_csv_sha256']} rows={s['correctness']['csv_rows_incl_header']}")

    shutil.rmtree(out_dir, ignore_errors=True)
    # Clean the raw pose NPZ the current (pre-refactor) pipeline drops under pose_data/ —
    # gitignored scratch, but large. Only remove if we created the tree.
    if not pose_data_existed:
        shutil.rmtree(ROOT / "pose_data", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
