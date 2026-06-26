#!/usr/bin/env python3
"""Streaming-pipeline benchmark + correctness harness for the RallyClip release path.

This is the measurement gate for the `perf/streaming-pipeline` optimization loop. It drives
the *real* data-handoff stages (pose -> preprocess -> features) and reports the metrics the
loop optimizes against:

  * per-stage wall time (pose / preprocess / features)
  * intermediate serialization time + bytes (np.savez_compressed / np.load on the
    tmp NPZ files) -- this is the disk-IO the streaming refactor removes
  * peak RSS delta during the run -- the memory the streaming/chunking refactor must bound

Stub mode (default) uses synthetic detections + a fake YOLO model so runs are fast,
deterministic, and length-scalable via --frames. That lets the loop prove two things:
  1. serialization time/bytes -> ~0 once stages hand off in memory
  2. peak RSS stays bounded (flat vs --frames) once the pipeline streams/chunks

Synthetic frames are intentionally tiny so the measured peak RSS reflects the accumulated
pose/preprocessed/feature *data* (the thing we optimize), not the decoded video buffers.

Correctness: the harness hashes the produced feature matrix (+ targets) and the derived
segments. The refactor must keep both hashes identical to the golden baseline.

Usage:
  python scripts/perf/bench_pipeline.py --frames 6000 --json out.json
  python scripts/perf/bench_pipeline.py --frames 27000            # ~30 min @ 15fps
  python scripts/perf/bench_pipeline.py --frames 2000 --pretty
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:  # real RSS (includes numpy buffers); falls back to rusage if missing
    import psutil

    _PROC = psutil.Process()
except Exception:  # pragma: no cover
    psutil = None
    _PROC = None


# --------------------------------------------------------------------------------------
# Memory + serialization instrumentation
# --------------------------------------------------------------------------------------
def _rss_bytes() -> int:
    if _PROC is not None:
        return int(_PROC.memory_info().rss)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru if sys.platform == "darwin" else ru * 1024)  # darwin: bytes, linux: KB


class RSSSampler:
    """Background sampler for peak RSS during a measured region."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.base = 0
        self.peak = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self) -> "RSSSampler":
        self.base = _rss_bytes()
        self.peak = self.base
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            r = _rss_bytes()
            if r > self.peak:
                self.peak = r

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)
        r = _rss_bytes()
        if r > self.peak:
            self.peak = r

    @property
    def peak_delta_mb(self) -> float:
        return (self.peak - self.base) / 1e6


# Cumulative serialization accounting, populated by the patched numpy IO below.
SER = {"write_s": 0.0, "write_mb": 0.0, "read_s": 0.0, "read_mb": 0.0, "writes": 0, "reads": 0}


def _install_numpy_io_probes() -> None:
    """Wrap numpy.savez_compressed / numpy.load to attribute serialization cost.

    The stage modules call ``np.savez_compressed`` / ``np.load`` at call time, so patching the
    attributes on the numpy module is enough for them to pick up the wrappers.
    """
    real_savez = np.savez_compressed
    real_load = np.load

    def _sizeof(file_arg) -> float:
        try:
            p = Path(file_arg)
            if p.suffix == "":
                p = p.with_suffix(".npz")
            return p.stat().st_size / 1e6 if p.exists() else 0.0
        except Exception:
            return 0.0

    def savez_compressed(file, *args, **kwds):
        t0 = time.perf_counter()
        out = real_savez(file, *args, **kwds)
        SER["write_s"] += time.perf_counter() - t0
        if isinstance(file, (str, Path)):
            SER["write_mb"] += _sizeof(file)
        SER["writes"] += 1
        return out

    def load(file, *args, **kwds):
        mb = _sizeof(file) if isinstance(file, (str, Path)) else 0.0
        t0 = time.perf_counter()
        out = real_load(file, *args, **kwds)
        SER["read_s"] += time.perf_counter() - t0
        SER["read_mb"] += mb
        SER["reads"] += 1
        return out

    np.savez_compressed = savez_compressed
    np.load = load


# --------------------------------------------------------------------------------------
# Synthetic, deterministic YOLO stub
# --------------------------------------------------------------------------------------
class _T:
    """Mimic a torch tensor's .detach().cpu().numpy() chain over a numpy array."""

    def __init__(self, v) -> None:
        self.v = np.asarray(v, dtype=np.float32)

    def detach(self) -> "_T":
        return self

    def cpu(self) -> "_T":
        return self

    def numpy(self) -> np.ndarray:
        return self.v


def _fake_result(i: int):
    """Two deterministic detections (near + far player) with frame-indexed drift."""
    dx = (i % 50) * 0.3
    near_box = np.array([100.0 + dx, 400.0, 140.0 + dx, 500.0], dtype=np.float32)
    far_box = np.array([600.0 - dx, 200.0, 630.0 - dx, 260.0], dtype=np.float32)
    near_kp = np.stack([[100.0 + k + dx, 410.0 + 2 * k] for k in range(17)]).astype(np.float32)
    far_kp = np.stack([[600.0 + k - dx, 210.0 + 2 * k] for k in range(17)]).astype(np.float32)
    boxes = np.stack([near_box, far_box])
    kps = np.stack([near_kp, far_kp])
    box_conf = np.array([0.90, 0.70], dtype=np.float32)
    kp_conf = np.full((2, 17), 0.55, dtype=np.float32)
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=_T(boxes), conf=_T(box_conf)),
        keypoints=SimpleNamespace(xy=_T(kps), conf=_T(kp_conf)),
    )


class _FakeModel:
    """Stand-in for ultralytics YOLO: one fake result per frame in the batch."""

    def __init__(self) -> None:
        self.i = 0

    def predict(self, source=None, **kwargs):
        out = []
        for _ in source:
            out.append(_fake_result(self.i))
            self.i += 1
        return out

    def to(self, *_a, **_k):
        return self


def _synthetic_frames(n: int, fps: float):
    """Yield (tiny_frame, timestamp) so every frame maps to a processed target frame."""
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(n):
        yield frame, i / fps


# --------------------------------------------------------------------------------------
# Pipeline driver (stub mode)
# --------------------------------------------------------------------------------------
def run_stub(frames: int, fps: float, tmp_dir: Path, ref_w: int = 1280, ref_h: int = 720) -> dict:
    from extraction.pose_extractor import PoseExtractor
    from preprocessing.data_preprocessor import DataPreprocessor
    from features.feature_engineer import FeatureEngineer
    from infer import extract_segments_from_binary, gaussian_filter1d, hysteresis_threshold

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build a PoseExtractor without loading a real model.
    ex = PoseExtractor.__new__(PoseExtractor)
    ex.model_path = "yolov8n-pose.pt"
    ex.model_dir = None
    ex.imgsz = 960
    ex.device = "cpu"
    ex.batch_size = 16
    ex.model = _FakeModel()
    ex.frame_iterator_pyav = lambda _vp: _synthetic_frames(frames, fps)  # type: ignore[assignment]

    stage_time: dict[str, float] = {}

    pre = DataPreprocessor(screen_width=ref_w, screen_height=ref_h, save_court_masks=False)
    # Synthetic "all in-bounds" court mask (0 == kept). Passing this avoids triggering the
    # real CourtDetector/YOLO, keeping the stub deterministic and offline. Court filtering
    # itself is exercised, just with nothing culled.
    court_mask = np.zeros((ref_h, ref_w), dtype=np.uint8)
    fe = FeatureEngineer(screen_width=ref_w, screen_height=ref_h, target_fps=fps)
    # synthetic.mp4 can't be probed -> _source_frame_shape falls back to (ref_h, ref_w), i.e.
    # an identity rescale, exactly as the file path did inside preprocess_single_video.
    src_height, src_width, _ = pre._source_frame_shape("synthetic.mp4")

    # Mirrors the release path: the stages hand off in memory AND stream --
    # iter_pose_frames -> iter_preprocess_frames -> iter_build_features -- so pose_data and the
    # preprocessed records are produced-and-discarded one frame at a time (peak memory
    # O(batch + 1), not O(num_frames)). The bench still materializes the feature matrix because
    # it must hash it (features_sha256) and derive segments from a global op on it; the real
    # pipeline streams feature rows into inference too (see bench_real.py).
    t0 = time.perf_counter()
    pose_stream = ex.iter_pose_frames(
        video_path="synthetic.mp4",
        confidence_threshold=0.25,
        start_time_seconds=0,
        duration_seconds=int(frames / fps) + 2,
        target_fps=int(fps),
        imgsz=960,
        annotations_csv=None,
    )
    pre_stream = pre.iter_preprocess_frames(pose_stream, court_mask, src_width, src_height)
    feature_rows, feature_targets = [], []
    for feature_vector, target in fe.iter_build_features(pre_stream):
        feature_rows.append(feature_vector)
        feature_targets.append(target)
    if feature_rows:
        feats = np.array(feature_rows, dtype=np.float32)
        targets = np.array(feature_targets)
    else:
        feats = np.empty((0, fe.feature_vector_size), dtype=np.float32)
        targets = np.empty((0,))
    stage_time["stream"] = time.perf_counter() - t0

    # Deterministic stand-in for windowed model inference: a fixed projection of the
    # features. The streaming refactor does not touch inference, so this is sufficient to
    # detect any change in the upstream feature matrix via the segment hash.
    rng = np.random.default_rng(0)
    w = rng.standard_normal(feats.shape[1]).astype(np.float32) if feats.size else np.zeros(0, np.float32)
    if feats.size:
        z = feats @ w
        z = (z - z.mean()) / (z.std() + 1e-6)  # z-score so probs span [0,1] and cross thresholds
        probs = (1.0 / (1.0 + np.exp(-1.5 * z))).astype(np.float32)
        probs = gaussian_filter1d(probs, sigma=2.0)
        binary = hysteresis_threshold(probs, low=0.45, high=0.55, min_duration=int(fps))
        segments = extract_segments_from_binary(binary)
    else:
        segments = []

    return {
        "stage_time": stage_time,
        "features": feats,
        "targets": targets,
        "segments": segments,
    }


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=6000, help="processed frames to simulate (15fps: 6000=~6.7min)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--json", type=str, default=None, help="write metrics JSON to this path")
    ap.add_argument("--pretty", action="store_true", help="print a human-readable summary")
    ap.add_argument("--tmp", type=str, default=None, help="tmp dir (default: ephemeral under /tmp)")
    args = ap.parse_args()

    import os
    import tempfile

    os.environ.setdefault("RALLYCLIP_NO_TQDM", "1")
    _install_numpy_io_probes()

    # Warm heavy imports BEFORE the RSS-measured region so peak_rss_delta reflects pipeline
    # *data* growth, not one-time torch/onnx/numpy import cost.
    import extraction.pose_extractor  # noqa: F401
    import preprocessing.data_preprocessor  # noqa: F401
    import features.feature_engineer  # noqa: F401
    import infer  # noqa: F401

    cleanup = False
    if args.tmp:
        tmp_dir = Path(args.tmp)
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="rcbench_"))
        cleanup = True

    SER.update({"write_s": 0.0, "write_mb": 0.0, "read_s": 0.0, "read_mb": 0.0, "writes": 0, "reads": 0})
    wall0 = time.perf_counter()
    with RSSSampler() as rss:
        result = run_stub(args.frames, args.fps, tmp_dir)
    elapsed = time.perf_counter() - wall0

    feats = result["features"]
    metrics = {
        "mode": "stub",
        "frames": args.frames,
        "fps": args.fps,
        "stage_time_s": {k: round(v, 4) for k, v in result["stage_time"].items()},
        "handoff_time_s": round(sum(result["stage_time"].values()), 4),
        "serialization": {
            "write_s": round(SER["write_s"], 4),
            "write_mb": round(SER["write_mb"], 3),
            "read_s": round(SER["read_s"], 4),
            "read_mb": round(SER["read_mb"], 3),
            "total_s": round(SER["write_s"] + SER["read_s"], 4),
            "total_mb": round(SER["write_mb"] + SER["read_mb"], 3),
            "writes": SER["writes"],
            "reads": SER["reads"],
        },
        "peak_rss_delta_mb": round(rss.peak_delta_mb, 2),
        "elapsed_total_s": round(elapsed, 4),
        "correctness": {
            "features_shape": list(feats.shape),
            "features_sha256": _sha(feats) if feats.size else "empty",
            "targets_sha256": _sha(result["targets"]) if result["targets"].size else "empty",
            "num_segments": len(result["segments"]),
            "segments_sha256": hashlib.sha256(str(result["segments"]).encode()).hexdigest()[:16],
        },
    }

    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.pretty or not args.json:
        s = metrics
        print(f"frames={s['frames']}  total={s['elapsed_total_s']}s  handoff={s['handoff_time_s']}s")
        print(f"  stage_time_s    : {s['stage_time_s']}")
        print(f"  serialization   : total {s['serialization']['total_s']}s / {s['serialization']['total_mb']}MB "
              f"(write {s['serialization']['write_s']}s, read {s['serialization']['read_s']}s, "
              f"{s['serialization']['writes']}w/{s['serialization']['reads']}r)")
        print(f"  peak_rss_delta  : {s['peak_rss_delta_mb']} MB")
        print(f"  features        : shape={s['correctness']['features_shape']} sha={s['correctness']['features_sha256']}")
        print(f"  segments        : n={s['correctness']['num_segments']} sha={s['correctness']['segments_sha256']}")

    if cleanup:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
