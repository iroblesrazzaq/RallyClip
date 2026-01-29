from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from training.dataset.builder import DatasetBuilder, DatasetConfig
from training.dataset.splits import SplitConfig
from training.eval.checkpoint import evaluate_checkpoint
from training.eval.evaluator import SegmentEvalConfig
from training.features.builder import FeatureBuildConfig, FeatureBuilder
from training.io.videos import resolve_videos
from training.paths import (
    annotations_dir,
    datasets_dir,
    pose_features_dir,
    pose_preprocessed_dir,
    pose_raw_dir,
    raw_videos_dir,
    resolve_data_root,
    runs_dir,
)
from training.pose.yolo_hdf5 import YoloExtractConfig, YoloHdf5Extractor
from training.preprocess.preprocessor import Hdf5Preprocessor, PreprocessConfig
from training.train.loop import train as train_loop

logger = logging.getLogger(__name__)

DEFAULT_STEPS = ["extract", "preprocess", "features", "dataset", "train", "eval"]


def run_pipeline(config: Dict[str, Any], steps_override: Optional[Iterable[str]] = None) -> None:
    steps = list(steps_override) if steps_override is not None else list(config.get("steps", DEFAULT_STEPS))
    run_id = config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_id"] = run_id

    data_root = resolve_data_root(config)
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

    raw_dir = raw_videos_dir(data_root)
    ann_dir = annotations_dir(data_root)

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
    imgsz = int(yolo_cfg.get("imgsz", 1920))

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
            imgsz=imgsz,
        )
    )

    conf_tag = _format_conf(conf)
    model_tag = Path(model_path).name
    output_root = pose_raw_dir(data_root) / f"yolo={model_tag}" / f"conf={conf_tag}" / f"imgsz={imgsz}"

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

    raw_dir = raw_videos_dir(data_root)
    ann_dir = annotations_dir(data_root)

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
    imgsz = int(yolo_cfg.get("imgsz", 1920))
    fps = float(preprocess_cfg.get("target_fps", 15))

    output_root = (
        pose_preprocessed_dir(data_root)
        / f"yolo={model_tag}"
        / f"conf={conf_tag}"
        / f"imgsz={imgsz}"
        / f"fps={fps}"
    )

    start_time = float(extract_cfg.get("start_time", 0))
    duration = _parse_optional_float(extract_cfg.get("duration"))
    raw_root = pose_raw_dir(data_root) / f"yolo={model_tag}" / f"conf={conf_tag}" / f"imgsz={imgsz}"

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

    raw_dir = raw_videos_dir(data_root)
    ann_dir = annotations_dir(data_root)

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
    imgsz = int(yolo_cfg.get("imgsz", 1920))
    fps = float(preprocess_cfg.get("target_fps", 15))
    feature_set = features_cfg.get("feature_set", "v1")

    preproc_root = (
        pose_preprocessed_dir(data_root)
        / f"yolo={model_tag}"
        / f"conf={conf_tag}"
        / f"imgsz={imgsz}"
        / f"fps={fps}"
    )
    output_root = (
        pose_features_dir(data_root)
        / f"yolo={model_tag}"
        / f"conf={conf_tag}"
        / f"imgsz={imgsz}"
        / f"fps={fps}"
    )

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

    raw_dir = raw_videos_dir(data_root)
    ann_dir = annotations_dir(data_root)

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
    imgsz = int(yolo_cfg.get("imgsz", 1920))
    fps = float(preprocess_cfg.get("target_fps", 15))
    feature_set = features_cfg.get("feature_set", "v1")

    feature_root = (
        pose_features_dir(data_root)
        / f"yolo={model_tag}"
        / f"conf={conf_tag}"
        / f"imgsz={imgsz}"
        / f"fps={fps}"
    )

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

    output_dir = datasets_dir(data_root) / config.get("run_id", "default")
    builder.build(feature_root, output_dir, videos, feature_set)


def _run_train(config: Dict[str, Any]) -> None:
    data_root = Path(config["data_root"])
    train_cfg = config.get("train", {})
    preprocess_cfg = config.get("preprocess", {})
    run_dir = runs_dir(data_root) / config.get("run_id", "default")
    dataset_dir = datasets_dir(data_root) / config.get("run_id", "default")

    segment_cfg = train_cfg.get("segment_eval", {})
    train_cfg["segment_eval"] = SegmentEvalConfig(
        low=float(segment_cfg.get("low", 0.45)),
        high=float(segment_cfg.get("high", 0.8)),
        sigma=float(segment_cfg.get("sigma", 1.5)),
        min_dur_sec=float(segment_cfg.get("min_dur_sec", 0.5)),
    )
    train_cfg["fps"] = float(preprocess_cfg.get("target_fps", 15))
    train_loop(dataset_dir, run_dir, train_cfg)


def _run_eval(config: Dict[str, Any]) -> None:
    data_root = Path(config["data_root"])
    train_cfg = config.get("train", {})
    preprocess_cfg = config.get("preprocess", {})

    run_dir = runs_dir(data_root) / config.get("run_id", "default")
    dataset_dir = datasets_dir(data_root) / config.get("run_id", "default")
    checkpoint_path = run_dir / "checkpoints" / "best.pth"
    test_path = dataset_dir / "test.h5"

    seg_cfg = train_cfg.get("segment_eval", {})
    segment_cfg = SegmentEvalConfig(
        low=float(seg_cfg.get("low", 0.45)),
        high=float(seg_cfg.get("high", 0.8)),
        sigma=float(seg_cfg.get("sigma", 1.5)),
        min_dur_sec=float(seg_cfg.get("min_dur_sec", 0.5)),
    )

    metrics, loss = evaluate_checkpoint(
        checkpoint_path,
        test_path,
        device_str=train_cfg.get("device"),
        threshold=float(train_cfg.get("threshold", 0.5)),
        segment_cfg=segment_cfg,
        fps=float(preprocess_cfg.get("target_fps", 15)),
        pos_weight=float(train_cfg.get("pos_weight", 3.0)),
    )
    logger.info("Test loss: %.4f", loss)
    logger.info("Test metrics: %s", metrics)




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
