"""Spike: CoreML EP vs CPU EP for RallyClip's shipped ONNX models.

Measures throughput and per-output numeric divergence on real video frames.
Divergence matters because golden parity is byte-exact on CPU; any accelerated
path must either match or ship with a documented tolerance.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf")
sys.path.insert(0, str(REPO / "src"))

import onnxruntime as ort  # noqa: E402

from extraction.yolo_onnx_runner import letterbox  # noqa: E402

POSE_ONNX = REPO / "models/rallyclip_v0.3.1/yolov8n-pose-960-dynamic.onnx"
LSTM_ONNX = REPO / "models/rallyclip_v0.3.1/model.onnx"
SOURCE = Path.home() / "Library/Application Support/RallyClip/library/20260705-012328-b3235b/source.mp4"
N_FRAMES = 40
IMGSZ = 960

results = {}


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


def preprocess(frame):
    # letterbox already returns NCHW float32 RGB in [0,1]
    tensor, _, _ = letterbox(frame, new_shape=IMGSZ)
    return tensor


def bench(model_path, feeds_list, providers, warmup=3):
    sess = ort.InferenceSession(str(model_path), providers=providers)
    used = sess.get_providers()
    for feeds in feeds_list[:warmup]:
        sess.run(None, feeds)
    t0 = time.perf_counter()
    outs = [sess.run(None, feeds) for feeds in feeds_list]
    dt = time.perf_counter() - t0
    return outs, dt, used


print("decoding frames...", flush=True)
frames = decode_frames(N_FRAMES)
pose_inputs = [{"images": preprocess(f)} for f in frames]
print(f"{len(pose_inputs)} frames prepared", flush=True)

# --- pose model ---
cpu_out, cpu_dt, cpu_prov = bench(POSE_ONNX, pose_inputs, ["CPUExecutionProvider"])
print(f"pose CPU: {cpu_dt:.2f}s ({len(pose_inputs)/cpu_dt:.1f} fps) {cpu_prov}", flush=True)

coreml_out, coreml_dt, coreml_prov = bench(
    POSE_ONNX, pose_inputs, ["CoreMLExecutionProvider", "CPUExecutionProvider"]
)
print(f"pose CoreML: {coreml_dt:.2f}s ({len(pose_inputs)/coreml_dt:.1f} fps) {coreml_prov}", flush=True)

diffs = []
for a, b in zip(cpu_out, coreml_out):
    for x, y in zip(a, b):
        diffs.append(float(np.max(np.abs(x - y))))
pose_max_diff = max(diffs)
print(f"pose max abs output diff CPU vs CoreML: {pose_max_diff:.6g}", flush=True)

results["pose"] = {
    "cpu_s": round(cpu_dt, 3),
    "coreml_s": round(coreml_dt, 3),
    "speedup": round(cpu_dt / coreml_dt, 2),
    "coreml_providers_used": coreml_prov,
    "max_abs_diff": pose_max_diff,
    "n_frames": len(pose_inputs),
    "imgsz": IMGSZ,
}

# --- LSTM model ---
lstm_sess = ort.InferenceSession(str(LSTM_ONNX), providers=["CPUExecutionProvider"])
inp = lstm_sess.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
print(f"lstm input {inp.name} {inp.shape} -> {shape}", flush=True)
rng = np.random.default_rng(0)
lstm_inputs = [{inp.name: rng.standard_normal(shape, dtype=np.float32)} for _ in range(200)]

lc_out, lc_dt, _ = bench(LSTM_ONNX, lstm_inputs, ["CPUExecutionProvider"])
lm_out, lm_dt, lm_prov = bench(LSTM_ONNX, lstm_inputs, ["CoreMLExecutionProvider", "CPUExecutionProvider"])
lstm_diff = max(
    float(np.max(np.abs(x - y))) for a, b in zip(lc_out, lm_out) for x, y in zip(a, b)
)
print(f"lstm CPU {lc_dt:.2f}s CoreML {lm_dt:.2f}s diff {lstm_diff:.6g} {lm_prov}", flush=True)
results["lstm"] = {
    "cpu_s": round(lc_dt, 3),
    "coreml_s": round(lm_dt, 3),
    "speedup": round(lc_dt / lm_dt, 2),
    "coreml_providers_used": lm_prov,
    "max_abs_diff": lstm_diff,
    "n_runs": len(lstm_inputs),
}

out_path = Path(__file__).parent / "spike_coreml_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"wrote {out_path}", flush=True)
