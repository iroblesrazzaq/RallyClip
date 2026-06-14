"""Ad-hoc YOLOv8n-pose latency benchmark: MPS FP32 vs FP16 (+ CPU baseline).

Measures the real contract model (yolov8n-pose.pt @ imgsz=960) the RallyClip
pipeline uses. Synthetic frames are fine for latency: GPU/NPU compute depends on
imgsz/batch/dtype, not pixel content. Proper MPS warmup + synchronize included.
"""
import time
import numpy as np
import torch
from ultralytics import YOLO

IMGSZ = 960          # v0.3.1 contract
CONF = 0.25          # v0.3.1 contract
MODEL = "yolov8n-pose.pt"
H, W = 720, 1280     # manifest screen size


def make_frames(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(n)]


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def bench(device, half, batch, iters, warmup):
    model = YOLO(MODEL)
    model.to(device)
    frames = make_frames(batch)

    # Warmup (compiles Metal kernels; first calls are wildly slow on MPS).
    for _ in range(warmup):
        model.predict(frames, device=device, half=half, imgsz=IMGSZ,
                      batch=batch, conf=CONF, verbose=False)
    sync(device)

    speeds = []
    t0 = time.perf_counter()
    for _ in range(iters):
        r = model.predict(frames, device=device, half=half, imgsz=IMGSZ,
                          batch=batch, conf=CONF, verbose=False)
        speeds.append(r[0].speed)
    sync(device)
    t1 = time.perf_counter()

    wall_ms_per_frame = (t1 - t0) * 1000.0 / (iters * batch)
    inf = float(np.mean([s["inference"] for s in speeds]))
    pre = float(np.mean([s["preprocess"] for s in speeds]))
    post = float(np.mean([s["postprocess"] for s in speeds]))
    param_dtype = next(model.model.parameters()).dtype
    return {
        "device": device, "half_req": half, "batch": batch,
        "wall_ms_per_frame": wall_ms_per_frame,
        "fps": 1000.0 / wall_ms_per_frame,
        "ultra_pre_ms": pre, "ultra_inf_ms": inf, "ultra_post_ms": post,
        "param_dtype": str(param_dtype),
    }


def main():
    print(f"torch {torch.__version__} | mps_avail={torch.backends.mps.is_available()} "
          f"| model={MODEL} imgsz={IMGSZ} src={W}x{H}\n")

    configs = [
        ("mps", False, 1, 60, 12),
        ("mps", False, 8, 40, 8),
        ("mps", True,  1, 60, 12),
        ("mps", True,  8, 40, 8),
        ("cpu", False, 1, 15, 3),   # baseline: what the pipeline forces today
    ]

    rows = []
    for device, half, batch, iters, warmup in configs:
        try:
            res = bench(device, half, batch, iters, warmup)
            rows.append(res)
            print(f"[{device:3s} half={str(half):5s} bs={batch:2d}] "
                  f"{res['wall_ms_per_frame']:7.2f} ms/frame  "
                  f"{res['fps']:6.1f} fps  | "
                  f"ultra pre/inf/post = {res['ultra_pre_ms']:.1f}/"
                  f"{res['ultra_inf_ms']:.1f}/{res['ultra_post_ms']:.1f} ms  "
                  f"| weights={res['param_dtype']}")
        except Exception as e:
            print(f"[{device} half={half} bs={batch}] FAILED: {e}")

    # Summary deltas
    def find(device, half, batch):
        for r in rows:
            if r["device"] == device and r["half_req"] == half and r["batch"] == batch:
                return r
        return None

    print("\n--- summary ---")
    cpu = find("cpu", False, 1)
    m32_1 = find("mps", False, 1)
    m16_1 = find("mps", True, 1)
    m32_8 = find("mps", False, 8)
    m16_8 = find("mps", True, 8)
    if cpu and m32_1:
        print(f"MPS fp32 vs CPU (bs1):   {cpu['wall_ms_per_frame']/m32_1['wall_ms_per_frame']:.2f}x faster")
    if m32_1 and m16_1:
        print(f"FP16 vs FP32 on MPS (bs1): {m32_1['wall_ms_per_frame']/m16_1['wall_ms_per_frame']:.2f}x")
    if m32_8 and m16_8:
        print(f"FP16 vs FP32 on MPS (bs8): {m32_8['wall_ms_per_frame']/m16_8['wall_ms_per_frame']:.2f}x")
    if m32_1 and m32_8:
        print(f"batch8 vs batch1 FP32 MPS: {m32_1['wall_ms_per_frame']/m32_8['wall_ms_per_frame']:.2f}x")
    if cpu and m16_8:
        print(f"Best MPS (fp16 bs8) vs CPU baseline: {cpu['wall_ms_per_frame']/m16_8['wall_ms_per_frame']:.2f}x faster")


if __name__ == "__main__":
    main()
