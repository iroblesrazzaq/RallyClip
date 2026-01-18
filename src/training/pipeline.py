from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from training.dataset.builder import DatasetBuilder, DatasetConfig
from training.dataset.splits import SplitConfig
from training.features.builder import FeatureBuildConfig, FeatureBuilder
from training.io.videos import resolve_videos
from training.pose.yolo_hdf5 import YoloExtractConfig, YoloHdf5Extractor
from training.preprocess.preprocessor import Hdf5Preprocessor, PreprocessConfig

logger = logging.getLogger(__name__)

DEFAULT_STEPS = ["extract", "preprocess", "features", "dataset", "train", "eval"]


def run_pipeline(config: Dict[str, Any], steps_override: Optional[Iterable[str]] = None) -> None:
    steps = list(steps_override) if steps_override is not None else list(config.get("steps", DEFAULT_STEPS))
    run_id = config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_id"] = run_id

    data_root = Path(config.get("data_root", "data")).expanduser().resolve()
    config["data_root"] = str(data_root)

    logger.info("Run id: %s", run_id)
    logger.info("Steps: %s", ", ".join(steps))

    step_map = {
        "extract": _run_extract,
        "preprocess": _run_preprocess,
        "features": _run_features,
        "dataset": _run_dataset,
        "train": _run_train,
        "eval": _run_eval,
    }

    for step in steps:
        func = step_map.get(step)
        if func is None:
            raise ValueError(f"Unknown step: {step}")
        func(config)


def _run_extract(config: Dict[str, Any]) -> None:
    data_root = Path(config["data_root"])
    extract_cfg = config.get("extract", {})
    yolo_cfg = config.get("yolo", {})

    raw_dir = data_root / "raw_videos"
    ann_dir = data_root / "annotations"

    videos = resolve_videos(
        extract_cfg.get("mode", "annotated"),
        raw_dir,
        ann_dir,
        config.get("videos") or extract_cfg.get("videos"),
    )
    if not videos:
        raise ValueError("No videos to extract")

    conf = float(yolo_cfg.get("conf", 0.25))
    model_path = str(yolo_cfg.get("model", "yolov8s-pose.pt"))
    model_dir = yolo_cfg.get("model_dir", "models")

    start_time = float(extract_cfg.get("start_time", 0))
    duration = _parse_optional_float(extract_cfg.get("duration"))
    overwrite = bool(config.get("overwrite_all") or extract_cfg.get("overwrite", False))

    extractor = YoloHdf5Extractor(
        YoloExtractConfig(
            model_path=model_path,
            conf=conf,
            model_dir=str(model_dir) if model_dir else None,
            device=yolo_cfg.get("device"),
            batch_size=yolo_cfg.get("batch_size"),
        )
    )

    conf_tag = _format_conf(conf)
    model_tag = Path(model_path).name
    output_root = data_root / "pose_data" / "raw" / f"yolo={model_tag}" / f"conf={conf_tag}"

    for video in videos:
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = raw_dir / video
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        out_name = f"{video_path.stem}__start{_format_time(start_time)}__dur{_format_time(duration)}.h5"
        output_path = output_root / out_name
        extractor.extract(
            video_path=video_path,
            output_path=output_path,
            start_time=start_time,
            duration=duration,
            overwrite=overwrite,
            resume=bool(extract_cfg.get("resume", True)),
        )


def _run_preprocess(config: Dict[str, Any]) -> None:
    data_root = Path(config["data_root"])
    preprocess_cfg = config.get("preprocess", {})
    court_cfg = config.get("court", {})
    extract_cfg = config.get("extract", {})
    yolo_cfg = config.get("yolo", {})

    raw_dir = data_root / "raw_videos"
    ann_dir = data_root / "annotations"

    videos = resolve_videos(
        preprocess_cfg.get("mode", "annotated"),
        raw_dir,
        ann_dir,
        config.get("videos") or preprocess_cfg.get("videos"),
    )
    if not videos:
        raise ValueError("No videos to preprocess")

    conf = float(yolo_cfg.get("conf", 0.25))
    model_tag = Path(str(yolo_cfg.get("model", "yolov8s-pose.pt"))).name
    conf_tag = _format_conf(conf)
    fps = float(preprocess_cfg.get("target_fps", 15))

    output_root = data_root / "pose_data" / "preprocessed" / f"yolo={model_tag}" / f"conf={conf_tag}" / f"fps={fps}"

    start_time = float(extract_cfg.get("start_time", 0))
    duration = _parse_optional_float(extract_cfg.get("duration"))
    raw_root = data_root / "pose_data" / "raw" / f"yolo={model_tag}" / f"conf={conf_tag}"

    preprocessor = Hdf5Preprocessor(
        PreprocessConfig(
            target_fps=fps,
            save_court_masks=bool(preprocess_cfg.get("save_court_masks", False)),
            court_model_path=court_cfg.get("model_path", "yolov8s.pt"),
            court_target_time=int(court_cfg.get("target_time", 60)),
            court_force=bool(court_cfg.get("force", False)),
        )
    )

    overwrite = bool(config.get("overwrite_all") or preprocess_cfg.get("overwrite", False))

    for video in videos:
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = raw_dir / video
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        raw_h5 = raw_root / f"{video_path.stem}__start{_format_time(start_time)}__dur{_format_time(duration)}.h5"
        if not raw_h5.exists():
            raise FileNotFoundError(f"Raw HDF5 not found: {raw_h5}")

        annotations_path = ann_dir / f"{video_path.name}.json"
        output_path = output_root / f"{video_path.stem}__fps{fps}.h5"
        preprocessor.preprocess(
            data_root=data_root,
            raw_h5_path=raw_h5,
            video_path=video_path,
            annotations_path=annotations_path,
            output_path=output_path,
            overwrite=overwrite,
        )


def _run_features(config: Dict[str, Any]) -> None:
    data_root = Path(config["data_root"])
    features_cfg = config.get("features", {})
    preprocess_cfg = config.get("preprocess", {})
    yolo_cfg = config.get("yolo", {})

    raw_dir = data_root / "raw_videos"
    ann_dir = data_root / "annotations"

    videos = resolve_videos(
        features_cfg.get("mode", "annotated"),
        raw_dir,
        ann_dir,
        config.get("videos") or features_cfg.get("videos"),
    )
    if not videos:
        raise ValueError("No videos to build features")

    conf = float(yolo_cfg.get("conf", 0.25))
    model_tag = Path(str(yolo_cfg.get("model", "yolov8s-pose.pt"))).name
    conf_tag = _format_conf(conf)
    fps = float(preprocess_cfg.get("target_fps", 15))
    feature_set = features_cfg.get("feature_set", "v1")

    preproc_root = data_root / "pose_data" / "preprocessed" / f"yolo={model_tag}" / f"conf={conf_tag}" / f"fps={fps}"
    output_root = data_root / "pose_data" / "features" / f"yolo={model_tag}" / f"conf={conf_tag}" / f"fps={fps}"

    overwrite = bool(config.get("overwrite_all") or features_cfg.get("overwrite", False))

    builder = FeatureBuilder(
        FeatureBuildConfig(
            feature_set=feature_set,
            target_fps=fps,
            overwrite=overwrite,
        )
    )

    for video in videos:
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = raw_dir / video
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        preproc_path = preproc_root / f"{video_path.stem}__fps{fps}.h5"
        if not preproc_path.exists():
            raise FileNotFoundError(f"Preprocessed HDF5 not found: {preproc_path}")

        output_path = output_root / f"{video_path.stem}__features__{feature_set}.h5"
        builder.build(preproc_path, output_path, overwrite=overwrite)


def _run_dataset(config: Dict[str, Any]) -> None:
    data_root = Path(config["data_root"])
    dataset_cfg = config.get("dataset", {})
    features_cfg = config.get("features", {})
    preprocess_cfg = config.get("preprocess", {})
    yolo_cfg = config.get("yolo", {})

    raw_dir = data_root / "raw_videos"
    ann_dir = data_root / "annotations"

    videos = resolve_videos(
        dataset_cfg.get("mode", "annotated"),
        raw_dir,
        ann_dir,
        config.get("videos") or dataset_cfg.get("videos"),
    )
    if not videos:
        raise ValueError("No videos to build dataset")

    conf = float(yolo_cfg.get("conf", 0.25))
    model_tag = Path(str(yolo_cfg.get("model", "yolov8s-pose.pt"))).name
    conf_tag = _format_conf(conf)
    fps = float(preprocess_cfg.get("target_fps", 15))
    feature_set = features_cfg.get("feature_set", "v1")

    feature_root = data_root / "pose_data" / "features" / f"yolo={model_tag}" / f"conf={conf_tag}" / f"fps={fps}"

    split_cfg = SplitConfig(
        strategy=dataset_cfg.get("split", {}).get("strategy", "hybrid"),
        seed=int(dataset_cfg.get("split", {}).get("seed", 1337)),
        val_ratio=float(dataset_cfg.get("split", {}).get("val_ratio", 0.1)),
        test_ratio=float(dataset_cfg.get("split", {}).get("test_ratio", 0.1)),
        test_videos=dataset_cfg.get("split", {}).get("test_videos", []),
        val_videos=dataset_cfg.get("split", {}).get("val_videos", []),
    )

    builder = DatasetBuilder(
        DatasetConfig(
            seq_len_seconds=float(dataset_cfg.get("seq_len_seconds", 20)),
            overlap_seconds=float(dataset_cfg.get("overlap_seconds", 10)),
            target_fps=fps,
            split=split_cfg,
        )
    )

    output_dir = data_root / "datasets" / config.get("run_id", "default")
    builder.build(feature_root, output_dir, videos, feature_set)


def _run_train(config: Dict[str, Any]) -> None:
    logger.info("TODO: training loop")


def _run_eval(config: Dict[str, Any]) -> None:
    logger.info("TODO: evaluation")


def _format_conf(conf: float) -> str:
    text = f"{conf:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _format_time(value: Optional[float]) -> str:
    if value is None:
        return "full"
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _parse_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    return float(value)
