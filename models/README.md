# RallyClip Model Artifacts

Shipped inference artifacts:

- `models/rallyclip_v0.5.0/` — **current default**. Dilated TCN with pointness + start/end heatmap heads; hybrid decode (`frame_startend_heatmap`).
- `models/rallyclip_v0.4.0/` — classic bidirectional LSTM, pointness + hysteresis. Kept as fallback.
- `models/rallyclip_v0.3.1/` — previous LSTM recipe (5 fps / 20 s windows).
- `models/rallyclip_v0.1.0_legacy/` — original legacy artifact (15 fps / 300-frame sequences).

Each artifact directory contains:

1. **`model.onnx`** — ONNX segmenter (LSTM logits, or TCN 3-head logits).
2. **`scaler.json`** — `StandardScaler` parameters (`mean` / `scale`).
3. **`manifest.json`** — contract, postprocess knobs, provenance.
4. Pose ONNX siblings (`yolov8n-pose-960-dynamic.onnx`, static 544×960).

To run a non-default artifact from the CLI, pass the files explicitly:

```bash
rallyclip \
  --model-path models/rallyclip_v0.4.0/model.onnx \
  --scaler-path models/rallyclip_v0.4.0/scaler.json \
  --video path/to/match.mp4
```
