"""Spike v4: static rect (544x960) export — same shape the runtime feeds today."""
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).parent
REPO = Path("/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip-perf")
sys.path.insert(0, str(REPO / "src"))

N_FRAMES = 40
SOURCE = Path.home() / "Library/Application Support/RallyClip/library/20260705-012328-b3235b/source.mp4"

rect_onnx = SCRATCH / "yolov8n-pose-544x960-static.onnx"
if not rect_onnx.exists():
    from ultralytics import YOLO

    model = YOLO(str(REPO / "models" / "yolov8n-pose.pt"))
    exported = model.export(format="onnx", imgsz=[544, 960], dynamic=False, simplify=True)
    Path(exported).rename(rect_onnx)
print(f"rect onnx: {rect_onnx}", flush=True)

import onnxruntime as ort  # noqa: E402
from extraction.yolo_onnx_runner import letterbox  # noqa: E402


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
inputs = []
for f in decode_frames(N_FRAMES):
    t = letterbox(f, new_shape=960)[0]
    assert t.shape == (1, 3, 544, 960), t.shape
    inputs.append({"images": t})
print(f"{len(inputs)} frames prepared (rect 544x960)", flush=True)


def bench(providers, warmup=3):
    sess = ort.InferenceSession(str(rect_onnx), providers=providers)
    for feeds in inputs[:warmup]:
        sess.run(None, feeds)
    t0 = time.perf_counter()
    outs = [sess.run(None, feeds) for feeds in inputs]
    return outs, time.perf_counter() - t0


results = {}
cpu_out, cpu_dt = bench([("CPUExecutionProvider", {})])
print(f"rect CPU: {cpu_dt:.2f}s ({len(inputs)/cpu_dt:.1f} fps)", flush=True)
results["cpu_rect"] = {"s": round(cpu_dt, 3), "fps": round(len(inputs) / cpu_dt, 1)}

for name, opts in {
    "coreml_rect_all": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
    "coreml_rect_ane": {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndNeuralEngine"},
}.items():
    outs, dt = bench([("CoreMLExecutionProvider", opts), ("CPUExecutionProvider", {})])
    conf_diffs = []
    for a, b in zip(cpu_out, outs):
        pa, pb = a[0][0], b[0][0]
        mask = pa[4] > 0.25
        if mask.any():
            conf_diffs.append(float(np.max(np.abs(pa[:, mask] - pb[:, mask]))))
    conf_diff = max(conf_diffs) if conf_diffs else 0.0
    print(f"{name}: {dt:.2f}s ({len(inputs)/dt:.1f} fps) {cpu_dt/dt:.2f}x diff {conf_diff:.4g}", flush=True)
    results[name] = {
        "s": round(dt, 3),
        "fps": round(len(inputs) / dt, 1),
        "speedup_vs_cpu": round(cpu_dt / dt, 2),
        "confident_det_max_abs_diff": conf_diff,
    }

(SCRATCH / "spike_coreml_v4_results.json").write_text(json.dumps(results, indent=2))
print("done", flush=True)
