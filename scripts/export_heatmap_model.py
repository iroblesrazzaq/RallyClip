#!/usr/bin/env python3
"""Export a trained TennisPointHeatmapTCN checkpoint to the shippable
ONNX + scaler.json artifact contract used by the production heatmap runtime.

Produces, in --out-dir:
  model.onnx    input "features" [1, seq_len, feature_dim] -> three named logit
                outputs: pointness_logit, start_heatmap_logit, end_heatmap_logit
                (raw logits; the runner applies sigmoid itself)
  scaler.json   {"mean": [...], "scale": [...]}  (StandardScaler params)

Verifies torch vs onnxruntime parity before keeping the files.
Does NOT write manifest.json (assembled separately with postprocess params).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.models.heatmap_tcn import TennisPointHeatmapTCN  # noqa: E402


OUTPUT_NAMES = ("pointness_logit", "start_heatmap_logit", "end_heatmap_logit")


def load_tcn(checkpoint: Path, input_size: int, hidden_size: int, levels: int, kernel_size: int, dropout: float, head: str) -> TennisPointHeatmapTCN:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = TennisPointHeatmapTCN(
        input_size=input_size,
        hidden_size=hidden_size,
        levels=levels,
        kernel_size=kernel_size,
        dropout=dropout,
        head=head,
        stem_hidden=None,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def export_onnx(model: TennisPointHeatmapTCN, out_path: Path, seq_len: int, feature_dim: int, opset: int) -> None:
    dummy = torch.zeros(1, seq_len, feature_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["features"],
        output_names=list(OUTPUT_NAMES),
        opset_version=opset,
        dynamic_axes=None,
        dynamo=False,  # legacy exporter; torch 2.9+ defaults to onnxscript
    )


def export_scaler(scaler_path: Path, out_path: Path) -> int:
    import joblib

    scaler = joblib.load(scaler_path)
    mean = np.asarray(scaler.mean_, dtype=np.float64).tolist()
    scale = np.asarray(scaler.scale_, dtype=np.float64).tolist()
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"mean": mean, "scale": scale}, handle)
    return len(mean)


def parity_check(model: TennisPointHeatmapTCN, onnx_path: Path, seq_len: int, feature_dim: int, atol: float) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, seq_len, feature_dim)).astype(np.float32)
    with torch.no_grad():
        torch_outs = model(torch.from_numpy(x))
        torch_stack = np.stack([t.cpu().numpy().reshape(-1) for t in torch_outs], axis=0)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got_names = [o.name for o in sess.get_outputs()]
    if got_names != list(OUTPUT_NAMES):
        raise SystemExit(f"ONNX output names {got_names} != {list(OUTPUT_NAMES)}")
    onnx_outs = sess.run(list(OUTPUT_NAMES), {sess.get_inputs()[0].name: x})
    onnx_stack = np.stack([np.asarray(o).reshape(-1) for o in onnx_outs], axis=0)
    max_abs = float(np.max(np.abs(torch_stack - onnx_stack)))
    if max_abs > atol:
        raise SystemExit(f"PARITY FAILED: max |torch-onnx| = {max_abs:.2e} > {atol:.0e}")
    return max_abs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scaler", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--feature-dim", type=int, default=362)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--head", type=str, default="mlp")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.out_dir / "model.onnx"
    scaler_out = args.out_dir / "scaler.json"

    model = load_tcn(
        args.checkpoint,
        input_size=args.feature_dim,
        hidden_size=args.hidden_size,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        head=args.head,
    )
    export_onnx(model, model_out, args.seq_len, args.feature_dim, args.opset)
    n = export_scaler(args.scaler, scaler_out)
    max_abs = parity_check(model, model_out, args.seq_len, args.feature_dim, args.atol)

    print(
        f"model.onnx  <- {args.checkpoint}  "
        f"(input [1,{args.seq_len},{args.feature_dim}] -> {', '.join(OUTPUT_NAMES)})"
    )
    print(f"scaler.json <- {args.scaler}  ({n} dims)")
    print(f"parity OK: max |torch-onnx| = {max_abs:.2e} (atol {args.atol:.0e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
