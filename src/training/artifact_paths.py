"""Shared artifact path construction for yolo=/conf=/imgsz=/fps= layouts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


def format_conf(conf: float) -> str:
    text = f"{conf:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def yolo_conf_imgsz_tags(config: Dict[str, Any]) -> tuple[str, str, int]:
    yolo_cfg = config.get("yolo", {}) if isinstance(config.get("yolo"), dict) else {}
    model = Path(str(yolo_cfg.get("model", "yolov8n-pose.pt")))
    conf_tag = format_conf(float(yolo_cfg.get("conf", 0.25)))
    if model.name == "manifest.json" or (model.is_dir() and (model / "manifest.json").exists()):
        from extraction.pose_backend import load_manifest

        meta = load_manifest(model)
        # Manifest-backed models pin identity to the artifact (name@sha8) and
        # own their imgsz; config imgsz is ignored, matching YoloHdf5Extractor.
        return meta.tag, conf_tag, meta.imgsz
    imgsz = int(yolo_cfg.get("imgsz", 1920))
    return model.name, conf_tag, imgsz


def target_fps_from_config(config: Dict[str, Any], default: float = 5.0) -> float:
    preprocess_cfg = config.get("preprocess", {}) if isinstance(config.get("preprocess"), dict) else {}
    return float(preprocess_cfg.get("target_fps", default))


def feature_set_from_config(config: Dict[str, Any], default: str = "v1") -> str:
    features_cfg = config.get("features", {}) if isinstance(config.get("features"), dict) else {}
    return str(features_cfg.get("feature_set", default))


def pose_branch(
    data_root: Path,
    *,
    kind: str,
    model_tag: str,
    conf_tag: str,
    imgsz: int,
    fps: Optional[float] = None,
) -> Path:
    from training.paths import pose_features_dir, pose_preprocessed_dir, pose_raw_dir

    if kind == "raw":
        root = pose_raw_dir(data_root) / f"yolo={model_tag}" / f"conf={conf_tag}" / f"imgsz={imgsz}"
    elif kind == "preprocessed":
        if fps is None:
            raise ValueError("fps required for preprocessed paths")
        root = (
            pose_preprocessed_dir(data_root)
            / f"yolo={model_tag}"
            / f"conf={conf_tag}"
            / f"imgsz={imgsz}"
            / f"fps={fps}"
        )
    elif kind == "features":
        if fps is None:
            raise ValueError("fps required for features paths")
        root = (
            pose_features_dir(data_root)
            / f"yolo={model_tag}"
            / f"conf={conf_tag}"
            / f"imgsz={imgsz}"
            / f"fps={fps}"
        )
    else:
        raise ValueError(f"Unknown pose branch kind: {kind}")
    return root


def feature_root_from_config(data_root: Path, config: Dict[str, Any]) -> Path:
    model_tag, conf_tag, imgsz = yolo_conf_imgsz_tags(config)
    fps = target_fps_from_config(config)
    return pose_branch(
        data_root,
        kind="features",
        model_tag=model_tag,
        conf_tag=conf_tag,
        imgsz=imgsz,
        fps=fps,
    )


def feature_root_from_run_config(data_root: Path, run_cfg: Dict[str, Any]) -> Path:
    """Resolve feature root from a run's saved config.json (not hardcoded regex)."""
    # Train loop stores a flattened train config; also accept full pipeline configs.
    yolo = run_cfg.get("yolo") if isinstance(run_cfg.get("yolo"), dict) else {}
    preprocess = run_cfg.get("preprocess") if isinstance(run_cfg.get("preprocess"), dict) else {}
    features = run_cfg.get("features") if isinstance(run_cfg.get("features"), dict) else {}

    model = yolo.get("model") or run_cfg.get("yolo_model") or "yolov8n-pose.pt"
    conf = yolo.get("conf", run_cfg.get("yolo_conf", 0.25))
    imgsz = yolo.get("imgsz", run_cfg.get("yolo_imgsz", run_cfg.get("imgsz", 1920)))
    fps = preprocess.get("target_fps", run_cfg.get("fps", 5.0))
    feature_set = features.get("feature_set", run_cfg.get("feature_set", "v1"))
    _ = feature_set  # used by caller for filename; root is fps-keyed

    synthetic = {
        "yolo": {"model": model, "conf": conf, "imgsz": imgsz},
        "preprocess": {"target_fps": fps},
        "features": {"feature_set": feature_set},
    }
    return feature_root_from_config(data_root, synthetic)


def features_filename(video_stem: str, feature_set: str = "v1") -> str:
    return f"{video_stem}__features__{feature_set}.h5"


ConfigLike = Dict[str, Any]
PathLike = Union[str, Path]
