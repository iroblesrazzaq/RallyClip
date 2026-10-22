from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from training.paths import DATA_ROOT_DEFAULT

_DEFAULTS: Dict[str, Any] = {
    "data_root": DATA_ROOT_DEFAULT,
    "steps": ["normalize", "extract", "preprocess", "features", "dataset", "train", "eval"],
    "wandb": {"enabled": False, "project": None, "entity": None},
    "videos": [],
    "overwrite_all": False,
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Dict[str, Any]:
    """Load one or more YAML configs (comma-separated), deep-merging left→right."""
    paths = [p.strip() for p in str(path).split(",") if p.strip()]
    if not paths:
        raise ValueError("No config path provided")

    config = deepcopy(_DEFAULTS)
    loaded_paths = []
    for raw in paths:
        cfg_path = Path(raw).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        config = _deep_merge(config, data)
        loaded_paths.append(str(cfg_path))

    config["config_path"] = loaded_paths[0]
    config["config_paths"] = loaded_paths
    return config


def resolve_yolo_model_path(config: Dict[str, Any]) -> str:
    yolo_cfg = config.get("yolo", {})
    model = str(yolo_cfg.get("model", "yolov8s-pose.pt"))
    model_path = Path(model)
    if model_path.is_absolute() or model_path.parent != Path("."):
        return str(model_path)
    model_dir = str(yolo_cfg.get("model_dir", "models"))
    return str(Path(model_dir) / model_path.name)


def resolve_court_model_path(config: Dict[str, Any]) -> str:
    court_cfg = config.get("court", {})
    explicit = court_cfg.get("model_path")
    if explicit:
        return str(explicit)
    return resolve_yolo_model_path(config)
