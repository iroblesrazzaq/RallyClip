"""Spike v2: CoreML EP MLProgram variants + detection-level divergence.

v1 showed NeuralNetwork-format CoreML EP is only 1.2x on pose with 0.25 raw
output diff. Here: try MLProgram/compute-unit options, and check whether the
divergence survives postprocess (boxes/keypoints after NMS), which is what
actually matters for segment output parity.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]  # docs/perf/coreml-spike -> repo root
sys.path.insert(0, str(REPO / "src"))

import onnxruntime as ort  # noqa: E402
from extraction.yolo_onnx_runner import letterbox  # noqa: E402

POSE_ONNX = REPO / "models/rallyclip_v0.3.1/yolov8n-pose-960-dynamic.onnx"
# Any real match video works; results in this directory came from a saved
# 145MB h264 1280x720 24fps library item on the dev machine.
SOURCE = Path(os.environ["RALLYCLIP_SPIKE_SOURCE"]) if "RALLYCLIP_SPIKE_SOURCE" in os.environ else None
if SOURCE is None or not SOURCE.exists():
    raise SystemExit("Set RALLYCLIP_SPIKE_SOURCE to a match video (mp4) to run this spike.")
N_FRAMES = 40
IMGSZ = 960


def decode_frames(n):
    import av

    frames = []
    with av.open(str(SOURCE)) as container:
        stream = container.streams.video[0]
        step = max(1, int(stream.frames / n) if stream.frames else 24)
        for i, frame in enumerate(container.decode(stream)):
            if i % step == 0:
                frames.append(frame.to_ndarray(format="bgr24"))
            if len(frames) >= n:
                break
    return frames


print("decoding frames...", flush=True)
inputs = [{"images": letterbox(f, new_shape=IMGSZ)[0]} for f in decode_frames(N_FRAMES)]
print(f"{len(inputs)} frames prepared", flush=True)


def bench(providers, warmup=3):
    t_load = time.perf_counter()
    sess = ort.InferenceSession(str(POSE_ONNX), providers=providers)
    load_s = time.perf_counter() - t_load
    for feeds in inputs[:warmup]:
        sess.run(None, feeds)
    t0 = time.perf_counter()
    outs = [sess.run(None, feeds) for feeds in inputs]
    dt = time.perf_counter() - t0
    return outs, dt, load_s


results = {}
cpu_out, cpu_dt, _ = bench([("CPUExecutionProvider", {})])
print(f"CPU: {cpu_dt:.2f}s ({len(inputs)/cpu_dt:.1f} fps)", flush=True)
results["cpu"] = {"s": round(cpu_dt, 3), "fps": round(len(inputs) / cpu_dt, 1)}

variants = {
    "coreml_default": {},
    "coreml_mlprogram": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
    "coreml_mlprogram_cpu_gpu": {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"},
}

for name, opts in variants.items():
    try:
        outs, dt, load_s = bench([("CoreMLExecutionProvider", opts), ("CPUExecutionProvider", {})])
    except Exception as exc:
        print(f"{name}: FAILED {exc}", flush=True)
        results[name] = {"error": str(exc)}
        continue
    max_diff = max(
        float(np.max(np.abs(x - y))) for a, b in zip(cpu_out, outs) for x, y in zip(a, b)
    )
    # relative divergence on confident detections: rows where CPU conf > 0.25
    rel_diffs = []
    for a, b in zip(cpu_out, outs):
        pa, pb = a[0][0], b[0][0]  # (56, 18900): 4 box + 1 conf + 51 kpts
        mask = pa[4] > 0.25
        if mask.any():
            rel_diffs.append(float(np.max(np.abs(pa[:, mask] - pb[:, mask]))))
    conf_diff = max(rel_diffs) if rel_diffs else 0.0
    print(
        f"{name}: {dt:.2f}s ({len(inputs)/dt:.1f} fps) load {load_s:.1f}s "
        f"max_diff {max_diff:.4g} confident-det diff {conf_diff:.4g}",
        flush=True,
    )
    results[name] = {
        "s": round(dt, 3),
        "fps": round(len(inputs) / dt, 1),
        "load_s": round(load_s, 1),
        "speedup_vs_cpu": round(cpu_dt / dt, 2),
        "max_abs_diff": max_diff,
        "confident_det_max_abs_diff": conf_diff,
    }

out_path = Path(__file__).parent / "spike_coreml_v2_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"wrote {out_path}", flush=True)
