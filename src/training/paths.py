from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

DATA_ROOT_DEFAULT = "data"

RAW_VIDEOS_DIRNAME = "raw_videos"
ANNOTATIONS_DIRNAME = "annotations"
POSE_DATA_DIRNAME = "pose_data"
POSE_RAW_DIRNAME = "raw"
POSE_PREPROCESSED_DIRNAME = "preprocessed"
POSE_FEATURES_DIRNAME = "features"
POSE_COURTS_DIRNAME = "courts"
DATASETS_DIRNAME = "datasets"
RUNS_DIRNAME = "runs"
VISUALIZATIONS_DIRNAME = "visualizations"


def resolve_data_root(config: Dict[str, Any], default: str = DATA_ROOT_DEFAULT) -> Path:
    return Path(config.get("data_root", default)).expanduser().resolve()


def raw_videos_dir(data_root: Path) -> Path:
    return data_root / RAW_VIDEOS_DIRNAME


def annotations_dir(data_root: Path) -> Path:
    return data_root / ANNOTATIONS_DIRNAME


def pose_data_dir(data_root: Path) -> Path:
    return data_root / POSE_DATA_DIRNAME


def pose_raw_dir(data_root: Path) -> Path:
    return pose_data_dir(data_root) / POSE_RAW_DIRNAME


def pose_preprocessed_dir(data_root: Path) -> Path:
    return pose_data_dir(data_root) / POSE_PREPROCESSED_DIRNAME


def pose_features_dir(data_root: Path) -> Path:
    return pose_data_dir(data_root) / POSE_FEATURES_DIRNAME


def pose_courts_dir(data_root: Path) -> Path:
    return pose_data_dir(data_root) / POSE_COURTS_DIRNAME


def datasets_dir(data_root: Path) -> Path:
    return data_root / DATASETS_DIRNAME


def runs_dir(data_root: Path) -> Path:
    return data_root / RUNS_DIRNAME


def visualizations_dir(data_root: Path) -> Path:
    return data_root / VISUALIZATIONS_DIRNAME
