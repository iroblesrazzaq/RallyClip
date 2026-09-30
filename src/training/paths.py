from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

DATA_ROOT_DEFAULT = "data"

VIDEOS_DIRNAME = "videos"
SOURCE_VIDEOS_DIRNAME = "source_videos"
ANNOTATIONS_DIRNAME = "annotations"
POSE_DATA_DIRNAME = "pose_data"
POSE_RAW_DIRNAME = "raw"
POSE_PREPROCESSED_DIRNAME = "preprocessed"
POSE_FEATURES_DIRNAME = "features"
POSE_COURTS_DIRNAME = "courts"
DATASETS_DIRNAME = "datasets"
RUNS_DIRNAME = "runs"
VISUALIZATIONS_DIRNAME = "visualizations"

# Video normalization contract (see src/training/normalize/). The tag is a path
# level for everything derived from normalized video, so a future contract
# (e.g. 1080p native) becomes a sibling tree instead of a silent collision.
NORM_WIDTH_DEFAULT = 1280
NORM_HEIGHT_DEFAULT = 720
NORM_FPS_DEFAULT = 5.0


def norm_tag(
    width: int = NORM_WIDTH_DEFAULT,
    height: int = NORM_HEIGHT_DEFAULT,
    fps: float = NORM_FPS_DEFAULT,
) -> str:
    return f"norm={int(width)}x{int(height)}@{fps:g}fps"


NORM_TAG_DEFAULT = norm_tag()


def resolve_data_root(config: Dict[str, Any], default: str = DATA_ROOT_DEFAULT) -> Path:
    return Path(config.get("data_root", default)).expanduser().resolve()


def raw_videos_dir(data_root: Path, norm: str = NORM_TAG_DEFAULT) -> Path:
    """Normalized (contract-tagged) videos — the pipeline's working video tree."""
    return data_root / VIDEOS_DIRNAME / norm


def source_videos_dir(data_root: Path) -> Path:
    return data_root / SOURCE_VIDEOS_DIRNAME


def annotations_dir(data_root: Path) -> Path:
    return data_root / ANNOTATIONS_DIRNAME


def pose_data_dir(data_root: Path, norm: str = NORM_TAG_DEFAULT) -> Path:
    return data_root / POSE_DATA_DIRNAME / norm


def pose_raw_dir(data_root: Path, norm: str = NORM_TAG_DEFAULT) -> Path:
    return pose_data_dir(data_root, norm) / POSE_RAW_DIRNAME


def pose_preprocessed_dir(data_root: Path, norm: str = NORM_TAG_DEFAULT) -> Path:
    return pose_data_dir(data_root, norm) / POSE_PREPROCESSED_DIRNAME


def pose_features_dir(data_root: Path, norm: str = NORM_TAG_DEFAULT) -> Path:
    return pose_data_dir(data_root, norm) / POSE_FEATURES_DIRNAME


def pose_courts_dir(data_root: Path, norm: str = NORM_TAG_DEFAULT) -> Path:
    return pose_data_dir(data_root, norm) / POSE_COURTS_DIRNAME


def datasets_dir(data_root: Path) -> Path:
    return data_root / DATASETS_DIRNAME


def runs_dir(data_root: Path) -> Path:
    return data_root / RUNS_DIRNAME


def visualizations_dir(data_root: Path) -> Path:
    return data_root / VISUALIZATIONS_DIRNAME
