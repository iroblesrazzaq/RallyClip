from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULTS: Dict[str, Any] = {
    "data_root": "data",
    "steps": ["extract", "preprocess", "features", "dataset", "train", "eval"],
    "wandb": {"enabled": False, "project": None, "entity": None},
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
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    config = _deep_merge(_DEFAULTS, data)
    config["config_path"] = str(cfg_path)
    return config
