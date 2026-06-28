# RallyClip Model Artifacts

Available exported inference artifacts:

- `models/rallyclip_v0.3.1/`
  - current default artifact
  - 5 fps / 20s / mirrored training recipe
- `models/rallyclip_v0.1.0_legacy/`
  - original legacy artifact
  - 15 fps / 300-frame sequence defaults

Each artifact directory contains:

1. **`model.onnx`** - ONNX-exported bidirectional LSTM model.
2. **`scaler.json`** - Serialized `StandardScaler` parameters in JSON format.
3. **`manifest.json`** - Artifact provenance, metrics, postprocessing defaults, and source-run metadata.

Historical tracked source assets for the legacy model are also preserved:

- `models/lstm_300_v0.1.pth`
- `models/scaler_300_v0.1.joblib`

To run a non-default artifact from the CLI, pass both files explicitly:

```bash
rallyclip \
  --model-path models/rallyclip_v0.1.0_legacy/model.onnx \
  --scaler-path models/rallyclip_v0.1.0_legacy/scaler.json \
  --video path/to/match.mp4
```
