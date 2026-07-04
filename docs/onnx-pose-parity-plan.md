# ONNX Pose Runner Parity Plan

Goal: replace `ultralytics.YOLO.predict()` inside `PoseExtractor` with a
torch-free ONNX Runtime runner — same input parameters, same output arrays —
validated first at the byte level, then at characterized numeric tolerances,
then end-to-end against the golden pipeline. Prior art lives in the sibling
repo `../YOLO-ONNX` (C++ ORT runner, benchmark harness, design notes).

## The contract to replicate (audited 2026-07-03)

The *only* Ultralytics surface RallyClip consumes
([pose_extractor.py:210-246](../src/extraction/pose_extractor.py#L210)):

```python
results = model.predict(
    source=batch_frames,      # list of BGR ndarrays (PyAV-decoded)
    verbose=False,
    device=self.device,       # cpu | mps | cuda (resolved via runtime.device)
    conf=confidence_threshold,
    imgsz=predict_imgsz,      # DEFAULT 1920 — not 640
    batch=self.batch_size,    # 1 on cpu, 16 otherwise
)
# per result, exactly four arrays:
res.boxes.xyxy        # float32 [N, 4]
res.boxes.conf        # float32 [N]
res.keypoints.xy      # float32 [N, 17, 2]
res.keypoints.conf    # float32 [N, 17]
```

Implicit Ultralytics defaults that the runner must mirror exactly:
`iou=0.7` (not the 0.45 typical elsewhere), `max_det=300`,
`agnostic_nms=False`, detections sorted by confidence descending
(player_assigner consumes them in order).

Production weights: `yolov8n-pose.pt` (raw-proposal ONNX export → runner must
do NMS). The YOLO26 retrain (2026-06-11, at parity with v8n) exports
end-to-end `[1, 300, 57]` with NMS inside the graph — no NMS matching needed.

## ⚠ Gotchas that invalidate naive comparisons

1. **imgsz=1920.** All YOLO-ONNX numbers so far (IoU 0.992, kpt err 0.42px)
   were measured at 640. The export must be at 1920 (or dynamic axes) and all
   parity runs re-measured there. Pixel-error tolerances scale ~3× vs 640.
2. **NMS semantics.** For the v8n raw export, the runner's NMS must replicate
   `iou=0.7`, class-aware=off-by-default behavior, and Ultralytics
   tie-breaking. This is the single most likely source of discrete
   (detection-count) mismatches. The YOLO26 e2e export sidesteps it entirely.
3. **Determinism controls for byte tests.** CPU only, ORT
   `intra_op_num_threads=1`, torch single-threaded, batch=1. MPS/CoreML EP
   comparisons are a separate (later) study — they will differ more.
4. **Shared frames, decoded once.** Compare on PNGs extracted once (the
   YOLO-ONNX harness already discovered separate video-decode paths cause
   false mismatches). Reuse its extracted-frame sets + frames from the
   RallyClip golden clip.
5. **Preprocessing bit-identity.** Letterbox = cv2 INTER_LINEAR resize,
   center pad 114, BGR→RGB, /255, NCHW fp32. Both sides must use the same
   cv2 version and the same rounding for scale/pad or byte comparison is
   dead on arrival at the input tensor.
6. **Batch letterbox.** Ultralytics letterboxes a batch to a common shape;
   with same-size video frames this is a no-op, but verify batch=16 vs
   batch=1 gives identical torch outputs before blaming the ONNX side.

## Experiment ladder

### Stage 0 — freeze the oracle
- Pinned env (record ultralytics/torch/ort/cv2 versions).
- Frame set: ~40 diverse frames from the YOLO-ONNX benchmark sets +
  ~40 frames sampled from `tests/fixtures/golden_cli/clip.mp4`, saved as PNG.
- Run Ultralytics `.pt` (CPU, deterministic, batch=1) → save the four arrays
  per frame as the oracle NPZ.

### Stage 1 — export + byte-level tests
- Export `yolov8n-pose.pt` → ONNX at imgsz=1920 (`nms=False` and `nms=True`
  variants); export the retrained YOLO26 e2e at 1920.
- **Test 1a (torch vs ORT, same graph semantics):** feed the *same
  preprocessed input tensor* (one shared numpy letterbox) to torch `.pt` and
  ORT `.onnx`; diff raw output tensors. Expected verdict: **not byte-equal**
  (different kernels/accumulation order); record max|Δ| (typically 1e-5–1e-3).
  This bounds what any wrapper can achieve.
- **Test 1b (ORT-python vs YOLO-ONNX C++, same .onnx, same ORT version,
  threads=1):** byte equality is *plausible* here if preprocessing is
  bit-identical. Any diff localizes to preproc or decode code, not the model.

### Stage 2 — decoded-output parity (the contract level)
Compare oracle vs runner on the four arrays, per frame:
- detection count match rate (target 1.000 at matched conf/iou),
- greedy box matching: matched IoU ≥ 0.95,
- keypoint px error (report mean/p95/max; interpret relative to 1920),
- keypoint/box confidence deltas,
- **order agreement** (conf-descending) — player_assigner sensitivity.
Include adversarial frames: near-threshold detections, near-tie NMS pairs,
zero-person frames (empty-array paths).

### Stage 3 — end-to-end golden parity
- Implement `OnnxPoseExtractor` with the identical constructor/iterator
  contract and inject it via `RuntimeDeps.PoseExtractor` (already injectable —
  no engine changes).
- Run the CLI on the golden clip both ways; compare:
  - segments CSV (byte-equal on same machine is the hope; the existing 0.25s
    cross-platform bar is the floor),
  - feature matrices max|Δ| and frame-probability curve max|Δ| — these
    localize drift between pose, features, and decode stages.

### Stage 4 — tolerance characterization (if not byte-equal, which is likely)
- Sweep over the real-video benchmark set; report distribution of segment
  boundary shifts vs oracle (P50/P95/max).
- Baseline for "acceptable": torch 2.7→2.12 alone shifts golden boundaries
  by one 0.2s hop (measured 2026-07-03). If ONNX-vs-torch drift is within the
  noise we already accept from dependency bumps, it passes.

### Stage 5 — productionize decision
- **Wrapper language:** recommend a pure-Python numpy+onnxruntime port of the
  C++ decode logic (`yolo_onnx_runner` per the design notes). RallyClip
  already ships onnxruntime for the LSTM; zero new build toolchain; the C++
  CLI remains the reference implementation + benchmark harness. pybind11
  binding of the C++ is the fallback if Python decode is too slow at 1920.
- **Model choice:** v8n-onnx (drop-in weights, NMS risk) vs YOLO26 e2e
  (NMS-free decode, retrained at parity) — decide on Stage 2/4 data.
- Device map: `cpu`→CPUExecutionProvider now; `mps`→CoreML EP studied
  separately before enabling.
- Gate: golden parity + full e2e suite green + memory benchmark (expect
  ~150MB vs ~530MB class of win, re-measured at 1920), then remove
  torch/ultralytics from the runtime dependency set (keep as dev/export
  extra).

## Honest expectations

- Byte-equality vs Ultralytics-torch: **no** — cross-framework fp32 kernel
  differences make this structurally unattainable; Stage 1a exists to prove
  and quantify it, not to chase it.
- Byte-equality between our two ONNX paths (python/C++): achievable and
  worth enforcing — it pins the wrapper as a pure re-implementation.
- The bar that matters: decoded-contract parity (Stage 2 targets) and
  end-to-end segment parity within the noise floor we already accept (Stage
  4). That is also exactly what the existing golden + e2e suites can enforce
  in CI forever after the swap.
