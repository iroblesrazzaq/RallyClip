from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from runtime.assets import YOLO_SIZE_MAP, manifest_values, resolve_asset
from rallyclip_core.pipelines import pipeline_id_from_manifest_values


def resolve_default_artifacts(
    artifact_dir: Optional[str] = None,
) -> tuple[Path, Path, Path]:
    """Resolve bundled ONNX artifact paths for GUI defaults."""
    artifact_path = Path(artifact_dir).expanduser().resolve() if artifact_dir else None
    model_relatives = ["models/rallyclip_v0.5.0/model.onnx"]
    scaler_relatives = ["models/rallyclip_v0.5.0/scaler.json"]
    if artifact_path is not None:
        model_path = artifact_path / "model.onnx"
        scaler_path = artifact_path / "scaler.json"
        manifest_path = artifact_path / "manifest.json"
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Artifact dir missing model/scaler: {artifact_path}")
        return model_path.resolve(), scaler_path.resolve(), manifest_path.resolve()

    model_path = resolve_asset(
        None,
        env_var="RALLYCLIP_MODEL_PATH",
        relatives=model_relatives,
        description="RallyClip model artifact (ONNX)",
    )
    scaler_path = resolve_asset(
        None,
        env_var="RALLYCLIP_SCALER_PATH",
        relatives=scaler_relatives,
        description="RallyClip scaler artifact (JSON)",
    )
    manifest_path = model_path.parent / "manifest.json"
    return model_path, scaler_path, manifest_path


def build_gui_defaults(artifact_dir: Optional[str] = None) -> Dict[str, Any]:
    """Manifest-driven defaults for the GUI, aligned with CLI contract fields."""
    model_path, _scaler_path, manifest_path = resolve_default_artifacts(artifact_dir)
    manifest = manifest_values(model_path, manifest_path if manifest_path.exists() else None)

    required = (
        "fps",
        "seq_len",
        "overlap",
        "sigma",
        "low",
        "high",
        "min_dur_sec",
        "conf",
        "imgsz",
        "feature_set",
        "screen_width",
        "screen_height",
    )
    missing = [key for key in required if manifest.get(key) is None]
    if missing:
        raise KeyError(f"Manifest missing required fields: {', '.join(missing)}")

    yolo_file = str(manifest.get("yolo_model") or "yolov8n-pose.pt")
    yolo_size = next((k for k, v in YOLO_SIZE_MAP.items() if v == yolo_file), "nano")

    return {
        "write_csv": True,
        "segment_video": True,
        "yolo_size": yolo_size,
        "yolo_weights": yolo_file,
        "yolo_device": None,
        "output_name": None,
        "model_path": str(model_path),
        "artifact_dir": str(model_path.parent),
        "pipeline_id": pipeline_id_from_manifest_values(manifest),
        "fps": float(manifest["fps"]),
        "seq_len": int(manifest["seq_len"]),
        "overlap": int(manifest["overlap"]),
        "sigma": float(manifest["sigma"]),
        "low": float(manifest["low"]),
        "high": float(manifest["high"]),
        "min_dur_sec": float(manifest["min_dur_sec"]),
        "conf": float(manifest["conf"]),
        "imgsz": int(manifest["imgsz"]),
        "feature_set": str(manifest["feature_set"]),
        "screen_width": int(manifest["screen_width"]),
        "screen_height": int(manifest["screen_height"]),
        "start_time": 0,
        "duration": 999999,
    }
