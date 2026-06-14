from __future__ import annotations
from copy import deepcopy
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
import joblib
import numpy as np
import torch

from infer.inference import gaussian_filter1d, hysteresis_threshold, run_windowed_inference_average
from training.dataset.builder import DatasetBuilder, DatasetConfig
from training.dataset.splits import SplitConfig
from training.eval.checkpoint import evaluate_checkpoint
from training.eval.evaluator import SegmentEvalConfig
from training.features.builder import FeatureBuildConfig, FeatureBuilder
from training.io.config import resolve_court_model_path
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
from training.metrics.segment import (
    compute_time_point_classification_metrics,
    compute_time_segment_metrics,
    compute_weighted_segment_score,
)
from training.models.lstm import TennisPointLSTM
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


def run_sweep(config: Dict[str, Any]) -> None:
    sweep_cfg = config.get("sweep", {})
    if not isinstance(sweep_cfg, dict) or not bool(sweep_cfg.get("enabled", False)):
        raise ValueError("Sweep is disabled. Set sweep.enabled: true in config and run with --sweep.")

    data_root = resolve_data_root(config)
    raw_dir = raw_videos_dir(data_root)
    ann_dir = annotations_dir(data_root)

    loso_cfg = sweep_cfg.get("loso", {}) if isinstance(sweep_cfg.get("loso"), dict) else {}
    loso_mode = str(loso_cfg.get("mode", config.get("dataset", {}).get("mode", "annotated")))
    loso_videos_explicit = loso_cfg.get("videos")
    videos = resolve_videos(loso_mode, raw_dir, ann_dir, loso_videos_explicit)
    if not videos:
        raise ValueError("No videos found for LOSO sweep")
    videos = sorted(videos)
    required_stems = {Path(v).stem for v in videos}

    dataset_specs = _resolve_sweep_datasets(config, sweep_cfg, data_root, required_stems)
    if not dataset_specs:
        raise ValueError("No dataset configs resolved for sweep")

    fps_values = _as_float_list(
        sweep_cfg.get("fps_values"),
        [float(config.get("preprocess", {}).get("target_fps", 15))],
    )
    seq_values = _as_float_list(
        sweep_cfg.get("seq_len_seconds"),
        [float(config.get("dataset", {}).get("seq_len_seconds", 20))],
    )
    sweep_overlap_mode = str(sweep_cfg.get("overlap_mode", "half_seq_len"))
    sweep_overlap_fixed = sweep_cfg.get("overlap_seconds")
    pre_steps = [str(s) for s in sweep_cfg.get("steps_precompute", ["preprocess", "features"])]
    fold_steps = [str(s) for s in sweep_cfg.get("steps_per_fold", ["dataset", "train", "eval"])]
    resume = bool(sweep_cfg.get("resume", True))
    run_prefix = str(sweep_cfg.get("run_prefix", "cv"))
    val_ratio = float(loso_cfg.get("val_ratio", config.get("dataset", {}).get("split", {}).get("val_ratio", 0.1)))
    seed = int(loso_cfg.get("seed", config.get("dataset", {}).get("split", {}).get("seed", 1337)))
    cleanup_completed_datasets = bool(sweep_cfg.get("cleanup_completed_datasets", True))

    logger.info(
        "Sweep start: datasets=%d fps=%s seq_len=%s folds=%d",
        len(dataset_specs),
        fps_values,
        seq_values,
        len(videos),
    )

    for ds in dataset_specs:
        ds_name = str(ds["name"])
        ds_fps_values = _as_float_list(ds.get("fps_values"), fps_values)
        ds_seq_values = _as_float_list(ds.get("seq_len_seconds"), seq_values)
        for fps in ds_fps_values:
            pre_cfg = deepcopy(config)
            _apply_sweep_dataset(pre_cfg, ds)
            pre_cfg.setdefault("preprocess", {})["target_fps"] = float(fps)
            pre_cfg["run_id"] = f"{run_prefix}_{ds_name}_fps{_fmt_num(fps)}_prep"
            run_pipeline(pre_cfg, steps_override=pre_steps)

            for seq_len in ds_seq_values:
                for fold_idx, held_out_video in enumerate(videos, start=1):
                    fold_cfg = deepcopy(config)
                    _apply_sweep_dataset(fold_cfg, ds)
                    fold_cfg.setdefault("preprocess", {})["target_fps"] = float(fps)
                    fold_cfg.setdefault("dataset", {})["seq_len_seconds"] = float(seq_len)
                    fold_cfg["dataset"]["overlap_seconds"] = _resolve_sweep_overlap_seconds(
                        seq_len=seq_len,
                        overlap_mode=sweep_overlap_mode,
                        overlap_fixed=sweep_overlap_fixed,
                        base_overlap=float(config.get("dataset", {}).get("overlap_seconds", 10)),
                    )
                    fold_cfg.setdefault("dataset", {}).setdefault("split", {})
                    fold_cfg["dataset"]["split"]["strategy"] = "loso_temporal_val"
                    fold_cfg["dataset"]["split"]["seed"] = seed
                    fold_cfg["dataset"]["split"]["val_ratio"] = val_ratio
                    fold_cfg["dataset"]["split"]["test_ratio"] = 0.0
                    fold_cfg["dataset"]["split"]["test_videos"] = [held_out_video]

                    fold_slug = _slug(Path(held_out_video).stem, 36)
                    run_id = (
                        f"{run_prefix}_{ds_name}_fps{_fmt_num(fps)}_seq{_fmt_num(seq_len)}s_"
                        f"fold{fold_idx:02d}_{fold_slug}"
                    )
                    fold_cfg["run_id"] = run_id
                    fold_cfg["wandb"] = _enrich_wandb_sweep(
                        fold_cfg.get("wandb", {}),
                        run_prefix=run_prefix,
                        dataset_name=ds_name,
                        fps=fps,
                        seq_len=seq_len,
                        fold_idx=fold_idx,
                    )

                    best_ckpt = runs_dir(data_root) / run_id / "checkpoints" / "best.pth"
                    if resume and _is_fold_complete(
                        data_root=data_root,
                        run_id=run_id,
                        expected_epochs=int(fold_cfg.get("train", {}).get("epochs", config.get("train", {}).get("epochs", 0))),
                    ):
                        if cleanup_completed_datasets:
                            _cleanup_completed_fold_dataset(data_root, run_id)
                        logger.info("Skipping completed fold: %s", run_id)
                        continue

                    run_pipeline(fold_cfg, steps_override=fold_steps)
                    if cleanup_completed_datasets and _is_fold_complete(
                        data_root=data_root,
                        run_id=run_id,
                        expected_epochs=int(fold_cfg.get("train", {}).get("epochs", config.get("train", {}).get("epochs", 0))),
                    ):
                        _cleanup_completed_fold_dataset(data_root, run_id)

    logger.info("Sweep complete")


def run_postprocess_sweep(config: Dict[str, Any]) -> None:
    data_root = resolve_data_root(config)
    pp_cfg = config.get("postprocess_eval", {}) if isinstance(config.get("postprocess_eval"), dict) else {}
    sweep_cfg = config.get("sweep", {}) if isinstance(config.get("sweep"), dict) else {}

    run_prefix = str(pp_cfg.get("run_prefix") or sweep_cfg.get("run_prefix") or "")
    if not run_prefix:
        raise ValueError("postprocess_eval.run_prefix or sweep.run_prefix must be set")

    dataset_filters = {str(v) for v in pp_cfg.get("datasets", []) if str(v)}
    fps_filters = {float(v) for v in pp_cfg.get("fps_values", [])}
    seq_filters = {float(v) for v in pp_cfg.get("seq_len_seconds", [])}
    device_str = str(pp_cfg.get("device", "cpu"))
    iou_threshold = float(pp_cfg.get("iou_threshold", 0.5))
    point_well_coverage_threshold = float(pp_cfg.get("point_well_coverage_threshold", 0.9))
    weights_cfg = pp_cfg.get("weights", {}) if isinstance(pp_cfg.get("weights"), dict) else {}
    coverage_weight = float(weights_cfg.get("coverage", 0.4))
    segment_recall_weight = float(weights_cfg.get("segment_recall", 0.4))
    specificity_weight = float(weights_cfg.get("specificity", 0.2))

    rows = []
    for run_dir in sorted(runs_dir(data_root).glob(f"{run_prefix}_*")):
        if not run_dir.is_dir():
            continue
        parsed = _parse_sweep_run_id(run_dir.name, run_prefix)
        if parsed is None:
            continue
        if dataset_filters and parsed["dataset"] not in dataset_filters:
            continue
        if fps_filters and float(parsed["fps"]) not in fps_filters:
            continue
        if seq_filters and float(parsed["seq_len_seconds"]) not in seq_filters:
            continue
        result = _evaluate_postprocess_run(
            data_root=data_root,
            run_dir=run_dir,
            device_str=device_str,
            iou_threshold=iou_threshold,
            point_well_coverage_threshold=point_well_coverage_threshold,
            coverage_weight=coverage_weight,
            segment_recall_weight=segment_recall_weight,
            specificity_weight=specificity_weight,
        )
        if result is not None:
            rows.append({**parsed, **result})

    if not rows:
        logger.warning("No completed runs matched postprocess filters for prefix=%s", run_prefix)
        return

    logger.info(
        "Postprocess metric weights: coverage=%.3f segment_recall=%.3f specificity=%.3f iou_threshold=%.2f point_well_coverage_threshold=%.2f",
        coverage_weight,
        segment_recall_weight,
        specificity_weight,
        iou_threshold,
        point_well_coverage_threshold,
    )

    per_run_lines = ["dataset,fps,seq_len_seconds,fold,time_score,time_segment_recall,time_coverage,time_specificity,time_segment_f1,time_mean_iou"]
    for row in rows:
        per_run_lines.append(
            ",".join(
                [
                    row["dataset"],
                    _fmt_num(row["fps"]),
                    _fmt_num(row["seq_len_seconds"]),
                    str(int(row["fold"])),
                    f"{row['time_score']:.6f}",
                    f"{row['time_segment_recall']:.6f}",
                    f"{row['time_coverage']:.6f}",
                    f"{row['time_specificity']:.6f}",
                    f"{row['time_segment_f1']:.6f}",
                    f"{row['time_mean_iou']:.6f}",
                ]
            )
        )
    logger.info("Per-run postprocess results:\n%s", "\n".join(per_run_lines))

    grouped: Dict[tuple[str, float, float], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], float(row["fps"]), float(row["seq_len_seconds"]))
        grouped.setdefault(key, []).append(row)

    summary_lines = [
        "dataset,fps,seq_len_seconds,folds,mean_time_score,mean_time_segment_recall,mean_time_coverage,mean_time_specificity,mean_time_segment_f1,mean_time_iou"
    ]
    for (dataset_name, fps, seq_len), group_rows in sorted(grouped.items()):
        summary_lines.append(
            ",".join(
                [
                    dataset_name,
                    _fmt_num(fps),
                    _fmt_num(seq_len),
                    str(len(group_rows)),
                    f"{np.mean([r['time_score'] for r in group_rows]):.6f}",
                    f"{np.mean([r['time_segment_recall'] for r in group_rows]):.6f}",
                    f"{np.mean([r['time_coverage'] for r in group_rows]):.6f}",
                    f"{np.mean([r['time_specificity'] for r in group_rows]):.6f}",
                    f"{np.mean([r['time_segment_f1'] for r in group_rows]):.6f}",
                    f"{np.mean([r['time_mean_iou'] for r in group_rows]):.6f}",
                ]
            )
        )
    logger.info("Postprocess summary by combo:\n%s", "\n".join(summary_lines))

    point_run_lines = [
        "dataset,fps,seq_len_seconds,fold,total_true_points,total_pred_points,well_classified_points,cut_off_points,missed_points,false_detected_points,unmatched_predicted_points,well_classified_rate,cut_off_rate,missed_rate,false_detected_rate,unmatched_predicted_rate"
    ]
    for row in rows:
        point_run_lines.append(
            ",".join(
                [
                    row["dataset"],
                    _fmt_num(row["fps"]),
                    _fmt_num(row["seq_len_seconds"]),
                    str(int(row["fold"])),
                    str(int(row["total_true_points"])),
                    str(int(row["total_pred_points"])),
                    str(int(row["well_classified_points"])),
                    str(int(row["cut_off_points"])),
                    str(int(row["missed_points"])),
                    str(int(row["false_detected_points"])),
                    str(int(row["unmatched_predicted_points"])),
                    f"{row['well_classified_rate']:.6f}",
                    f"{row['cut_off_rate']:.6f}",
                    f"{row['missed_rate']:.6f}",
                    f"{row['false_detected_rate']:.6f}",
                    f"{row['unmatched_predicted_rate']:.6f}",
                ]
            )
        )
    logger.info("Per-run point classification results:\n%s", "\n".join(point_run_lines))

    point_summary_lines = [
        "dataset,fps,seq_len_seconds,folds,sum_true_points,sum_pred_points,sum_well_classified_points,sum_cut_off_points,sum_missed_points,sum_false_detected_points,sum_unmatched_predicted_points,mean_well_classified_rate,mean_cut_off_rate,mean_missed_rate,mean_false_detected_rate,mean_unmatched_predicted_rate"
    ]
    for (dataset_name, fps, seq_len), group_rows in sorted(grouped.items()):
        point_summary_lines.append(
            ",".join(
                [
                    dataset_name,
                    _fmt_num(fps),
                    _fmt_num(seq_len),
                    str(len(group_rows)),
                    str(int(sum(r["total_true_points"] for r in group_rows))),
                    str(int(sum(r["total_pred_points"] for r in group_rows))),
                    str(int(sum(r["well_classified_points"] for r in group_rows))),
                    str(int(sum(r["cut_off_points"] for r in group_rows))),
                    str(int(sum(r["missed_points"] for r in group_rows))),
                    str(int(sum(r["false_detected_points"] for r in group_rows))),
                    str(int(sum(r["unmatched_predicted_points"] for r in group_rows))),
                    f"{np.mean([r['well_classified_rate'] for r in group_rows]):.6f}",
                    f"{np.mean([r['cut_off_rate'] for r in group_rows]):.6f}",
                    f"{np.mean([r['missed_rate'] for r in group_rows]):.6f}",
                    f"{np.mean([r['false_detected_rate'] for r in group_rows]):.6f}",
                    f"{np.mean([r['unmatched_predicted_rate'] for r in group_rows]):.6f}",
                ]
            )
        )
    logger.info("Point classification summary by combo:\n%s", "\n".join(point_summary_lines))


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
    sampling_mode = str(extract_cfg.get("sampling_mode", "full_then_downsample"))
    sample_fps = _parse_optional_float(extract_cfg.get("sample_fps"))
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
        out_name = _raw_h5_filename(video_path.stem, start_time, duration, sampling_mode, sample_fps)
        output_path = output_root / out_name
        extractor.extract(
            video_path=video_path,
            output_path=output_path,
            start_time=start_time,
            duration=duration,
            sampling_mode=sampling_mode,
            sample_fps=sample_fps,
            overwrite=overwrite,
            resume=bool(extract_cfg.get("resume", True)),
        )


def _run_preprocess(config: Dict[str, Any]) -> None:
    data_root = Path(config["data_root"])
    preprocess_cfg = config.get("preprocess", {})
    court_cfg = config.get("court", {})
    extract_cfg = config.get("extract", {})
    yolo_cfg = config.get("yolo", {})
    court_model_path = resolve_court_model_path(config)

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
    sampling_mode = str(extract_cfg.get("sampling_mode", "full_then_downsample"))
    sample_fps = _parse_optional_float(extract_cfg.get("sample_fps"))
    raw_root = pose_raw_dir(data_root) / f"yolo={model_tag}" / f"conf={conf_tag}" / f"imgsz={imgsz}"

    preprocessor = Hdf5Preprocessor(
        PreprocessConfig(
            target_fps=fps,
            save_court_masks=bool(preprocess_cfg.get("save_court_masks", False)),
            court_model_path=court_model_path,
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

        raw_h5 = raw_root / _raw_h5_filename(video_path.stem, start_time, duration, sampling_mode, sample_fps)
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
    built_dir = builder.build(feature_root, output_dir, videos, feature_set)
    if built_dir is not None:
        run_dir = runs_dir(data_root) / config.get("run_id", "default")
        run_dir.mkdir(parents=True, exist_ok=True)
        scaler_src = built_dir / "scaler.joblib"
        scaler_dst = run_dir / "scaler.joblib"
        if scaler_src.exists():
            shutil.copy2(scaler_src, scaler_dst)


def _run_train(config: Dict[str, Any]) -> None:
    data_root = Path(config["data_root"])
    train_cfg = dict(config.get("train", {}))
    preprocess_cfg = config.get("preprocess", {})
    dataset_cfg = config.get("dataset", {})
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
    train_cfg["run_id"] = config.get("run_id", "default")
    train_cfg["wandb"] = dict(config.get("wandb", {}))
    train_cfg["dataset"] = {
        "seq_len_seconds": dataset_cfg.get("seq_len_seconds"),
        "overlap_seconds": dataset_cfg.get("overlap_seconds"),
        "split": dict(dataset_cfg.get("split", {})),
    }
    train_loop(dataset_dir, run_dir, train_cfg)


def _run_eval(config: Dict[str, Any]) -> None:
    data_root = Path(config["data_root"])
    train_cfg = config.get("train", {})
    preprocess_cfg = config.get("preprocess", {})

    run_dir = runs_dir(data_root) / config.get("run_id", "default")
    dataset_dir = datasets_dir(data_root) / config.get("run_id", "default")
    checkpoint_path = run_dir / "checkpoints" / "best.pth"
    test_path = dataset_dir / "test.h5"
    if not test_path.exists():
        logger.info("No test split at %s (test_ratio=0?); skipping eval", test_path)
        return

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
    eval_path = run_dir / "eval.json"
    with eval_path.open("w", encoding="utf-8") as handle:
        json.dump({"loss": float(loss), "metrics": metrics}, handle, indent=2)
    logger.info("Test loss: %.4f", loss)
    logger.info("Test metrics: %s", metrics)




def _evaluate_postprocess_run(
    *,
    data_root: Path,
    run_dir: Path,
    device_str: str,
    iou_threshold: float,
    point_well_coverage_threshold: float,
    coverage_weight: float,
    segment_recall_weight: float,
    specificity_weight: float,
) -> Optional[Dict[str, float]]:
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoints" / "best.pth"
    dataset_dir = datasets_dir(data_root) / run_dir.name
    scaler_path = run_dir / "scaler.joblib"
    if not scaler_path.exists():
        scaler_path = dataset_dir / "scaler.joblib"
    metrics_path = run_dir / "metrics.jsonl"
    if not (config_path.exists() and checkpoint_path.exists() and scaler_path.exists() and metrics_path.exists()):
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        run_cfg = json.load(handle)

    dataset_cfg = run_cfg.get("dataset", {}) if isinstance(run_cfg.get("dataset"), dict) else {}
    split_cfg = dataset_cfg.get("split", {}) if isinstance(dataset_cfg.get("split"), dict) else {}
    held_out_videos = split_cfg.get("test_videos", [])
    if not held_out_videos:
        return None

    held_out_video = str(held_out_videos[0])
    fps = float(run_cfg.get("fps", 15.0))
    seq_len_seconds = float(dataset_cfg.get("seq_len_seconds", 10.0))
    overlap_seconds = float(dataset_cfg.get("overlap_seconds", seq_len_seconds / 2.0))
    imgsz = _parse_imgsz_from_run_id(run_dir.name)
    feature_root = (
        pose_features_dir(data_root)
        / "yolo=yolov8n-pose.pt"
        / "conf=0p25"
        / f"imgsz={imgsz}"
        / f"fps={fps}"
    )
    feature_path = feature_root / f"{Path(held_out_video).stem}__features__v1.h5"
    if not feature_path.exists():
        logger.warning("Skipping %s; feature file missing: %s", run_dir.name, feature_path)
        return None

    scaler = joblib.load(scaler_path)
    with h5py.File(feature_path, "r") as h5f:
        features = np.asarray(h5f["features"][:], dtype=np.float32)
        targets = np.asarray(h5f["targets"][:], dtype=np.int8)
        timestamps = np.asarray(h5f["timestamps"][:], dtype=np.float64)
    scaled = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape).astype(np.float32)

    model, device = _load_training_checkpoint(checkpoint_path, input_size=scaled.shape[-1], device_str=device_str)

    seq_len = max(1, int(round(seq_len_seconds * fps)))
    overlap = max(0, int(round(overlap_seconds * fps)))
    probs = run_windowed_inference_average(model, device, scaled, seq_len, overlap)

    segment_cfg = run_cfg.get("segment_eval", {}) if isinstance(run_cfg.get("segment_eval"), dict) else {}
    sigma = float(segment_cfg.get("sigma", 1.5))
    low = float(segment_cfg.get("low", 0.45))
    high = float(segment_cfg.get("high", 0.8))
    min_dur_frames = int(round(float(segment_cfg.get("min_dur_sec", 0.5)) * fps))

    smoothed = gaussian_filter1d(probs.astype(np.float32), sigma=sigma)
    pred_bin = hysteresis_threshold(smoothed, low=low, high=high, min_duration=min_dur_frames)
    metrics = compute_time_segment_metrics(
        targets.astype(int),
        pred_bin.astype(int),
        timestamps=timestamps,
        iou_threshold=iou_threshold,
    )
    point_metrics = compute_time_point_classification_metrics(
        targets.astype(int),
        pred_bin.astype(int),
        timestamps=timestamps,
        iou_threshold=iou_threshold,
        well_coverage_threshold=point_well_coverage_threshold,
    )
    score = compute_weighted_segment_score(
        metrics,
        segment_recall_weight=segment_recall_weight,
        coverage_weight=coverage_weight,
        specificity_weight=specificity_weight,
    )
    return {
        "time_score": float(score),
        "time_segment_recall": float(metrics.get("segment_recall", 0.0)),
        "time_coverage": float(metrics.get("coverage", 0.0)),
        "time_specificity": float(metrics.get("specificity", 0.0)),
        "time_segment_f1": float(metrics.get("segment_f1", 0.0)),
        "time_mean_iou": float(metrics.get("mean_iou", 0.0)),
        "total_true_points": int(point_metrics.get("total_true_points", 0)),
        "total_pred_points": int(point_metrics.get("total_pred_points", 0)),
        "well_classified_points": int(point_metrics.get("well_classified_points", 0)),
        "cut_off_points": int(point_metrics.get("cut_off_points", 0)),
        "missed_points": int(point_metrics.get("missed_points", 0)),
        "false_detected_points": int(point_metrics.get("false_detected_points", 0)),
        "unmatched_predicted_points": int(point_metrics.get("unmatched_predicted_points", 0)),
        "well_classified_rate": float(point_metrics.get("well_classified_rate", 0.0)),
        "cut_off_rate": float(point_metrics.get("cut_off_rate", 0.0)),
        "missed_rate": float(point_metrics.get("missed_rate", 0.0)),
        "false_detected_rate": float(point_metrics.get("false_detected_rate", 0.0)),
        "unmatched_predicted_rate": float(point_metrics.get("unmatched_predicted_rate", 0.0)),
    }


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


def _raw_h5_filename(
    stem: str,
    start_time: float,
    duration: Optional[float],
    sampling_mode: str,
    sample_fps: Optional[float],
) -> str:
    base = f"{stem}__start{_format_time(start_time)}__dur{_format_time(duration)}"
    if sampling_mode == "downsample_then_extract":
        if sample_fps is None:
            raise ValueError("sample_fps must be set when sampling_mode='downsample_then_extract'")
        return f"{base}__samplefps{_format_time(sample_fps)}.h5"
    return f"{base}.h5"


def _as_float_list(values: Any, default: List[float]) -> List[float]:
    if values is None:
        return list(default)
    if not isinstance(values, list):
        raise ValueError(f"Expected list for sweep values, got: {type(values).__name__}")
    out = [float(v) for v in values]
    if not out:
        raise ValueError("Sweep value list cannot be empty")
    return out


def _resolve_sweep_datasets(
    config: Dict[str, Any],
    sweep_cfg: Dict[str, Any],
    data_root: Path,
    required_stems: set[str],
) -> List[Dict[str, Any]]:
    ds_cfg = sweep_cfg.get("datasets", {})
    if not isinstance(ds_cfg, dict):
        ds_cfg = {}
    mode = str(ds_cfg.get("mode", "auto"))
    entries = ds_cfg.get("entries") if isinstance(ds_cfg.get("entries"), list) else []
    require_all = bool(ds_cfg.get("require_all_videos", True))
    auto_limit: Optional[int] = None

    if mode.startswith("auto") and mode != "auto":
        suffix = mode[len("auto"):]
        if suffix.isdigit():
            auto_limit = int(suffix)
            mode = "auto"

    if mode == "manual":
        specs = [_normalize_manual_dataset_entry(e, config) for e in entries]
        return specs

    if mode != "auto":
        raise ValueError(f"Unknown sweep.datasets.mode: {mode}")

    specs = _discover_raw_datasets(data_root, required_stems=required_stems, require_all=require_all)
    if entries:
        manual = [_normalize_manual_dataset_entry(e, config) for e in entries]
        manual_names = {m["name"] for m in manual}
        specs = [s for s in specs if s["name"] in manual_names]
    if auto_limit is not None:
        specs = specs[:auto_limit]
    return specs


def _normalize_manual_dataset_entry(entry: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Each sweep.datasets.entries item must be a mapping")
    yolo_base = dict(config.get("yolo", {}))
    yolo_base.update(dict(entry.get("yolo", {})))
    model = str(yolo_base.get("model", "yolov8s-pose.pt"))
    conf = float(yolo_base.get("conf", 0.25))
    imgsz = int(yolo_base.get("imgsz", 1920))
    name = str(entry.get("name") or f"{_slug(Path(model).stem, 20)}_conf{_fmt_num(conf)}_img{imgsz}")
    spec = {"name": _slug(name, 48), "yolo": yolo_base, "extract": dict(entry.get("extract", {}))}
    if "fps_values" in entry:
        spec["fps_values"] = list(entry.get("fps_values", []))
    if "seq_len_seconds" in entry:
        spec["seq_len_seconds"] = list(entry.get("seq_len_seconds", []))
    return spec


def _discover_raw_datasets(data_root: Path, required_stems: set[str], require_all: bool) -> List[Dict[str, Any]]:
    raw_root = pose_raw_dir(data_root)
    specs: List[Dict[str, Any]] = []
    if not raw_root.exists():
        return specs

    for yolo_dir in sorted(raw_root.glob("yolo=*")):
        if not yolo_dir.is_dir():
            continue
        model = yolo_dir.name.split("=", 1)[1]
        for conf_dir in sorted(yolo_dir.glob("conf=*")):
            if not conf_dir.is_dir():
                continue
            conf_tag = conf_dir.name.split("=", 1)[1]
            conf = float(conf_tag.replace("p", "."))
            for imgsz_dir in sorted(conf_dir.glob("imgsz=*")):
                if not imgsz_dir.is_dir():
                    continue
                try:
                    imgsz = int(imgsz_dir.name.split("=", 1)[1])
                except ValueError:
                    continue
                h5_files = sorted(imgsz_dir.glob("*.h5"))
                if not h5_files:
                    continue
                stems = {p.name.split("__start", 1)[0] for p in h5_files}
                if require_all and not required_stems.issubset(stems):
                    continue
                name = f"{_slug(Path(model).stem, 20)}_conf{_fmt_num(conf)}_img{imgsz}"
                specs.append(
                    {
                        "name": _slug(name, 48),
                        "yolo": {"model": model, "conf": conf, "imgsz": imgsz},
                        "extract": {},
                    }
                )
    return specs


def _apply_sweep_dataset(config: Dict[str, Any], dataset_spec: Dict[str, Any]) -> None:
    yolo_cfg = config.setdefault("yolo", {})
    yolo_cfg.update(dict(dataset_spec.get("yolo", {})))
    extract_cfg = config.setdefault("extract", {})
    extract_cfg.update(dict(dataset_spec.get("extract", {})))


def _enrich_wandb_sweep(
    wandb_cfg: Any,
    run_prefix: str,
    dataset_name: str,
    fps: float,
    seq_len: float,
    fold_idx: int,
) -> Dict[str, Any]:
    cfg = dict(wandb_cfg) if isinstance(wandb_cfg, dict) else {}
    if not bool(cfg.get("enabled", False)):
        return cfg
    cfg.setdefault("group", f"{run_prefix}_{dataset_name}_fps{_fmt_num(fps)}_seq{_fmt_num(seq_len)}s")
    tags = cfg.get("tags")
    tag_list = [str(t) for t in tags] if isinstance(tags, list) else []
    for tag in ("cv", "loso", f"fps_{_fmt_num(fps)}", f"seq_{_fmt_num(seq_len)}s", f"fold_{fold_idx:02d}"):
        if tag not in tag_list:
            tag_list.append(tag)
    cfg["tags"] = tag_list
    return cfg


def _fmt_num(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _slug(text: str, max_len: int) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not out:
        out = "x"
    return out[:max_len]


def _resolve_sweep_overlap_seconds(
    seq_len: float,
    overlap_mode: str,
    overlap_fixed: Any,
    base_overlap: float,
) -> float:
    if overlap_mode == "half_seq_len":
        return float(seq_len) / 2.0
    if overlap_mode == "fixed":
        if overlap_fixed in (None, "", "null"):
            raise ValueError("sweep.overlap_seconds must be set when sweep.overlap_mode='fixed'")
        return float(overlap_fixed)
    if overlap_mode == "base":
        return float(base_overlap)
    raise ValueError(f"Unknown sweep.overlap_mode: {overlap_mode}")


def _load_training_checkpoint(checkpoint_path: Path, input_size: int, device_str: str) -> tuple[torch.nn.Module, torch.device]:
    device = _resolve_requested_device(device_str)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    hidden_size = 128
    num_layers = 2
    bidirectional = False
    w_ih_l0 = state_dict.get("lstm.weight_ih_l0")
    if w_ih_l0 is not None:
        hidden_size = int(w_ih_l0.shape[0] // 4)
        input_size = int(w_ih_l0.shape[1])
    layer_ids = set()
    for key in state_dict.keys():
        match = re.match(r"lstm\.weight_ih_l(\d+)(?:_reverse)?$", key)
        if match:
            layer_ids.add(int(match.group(1)))
        if "_reverse" in key:
            bidirectional = True

    if layer_ids:
        num_layers = max(layer_ids) + 1

    model = TennisPointLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        return_logits=False,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def _resolve_requested_device(device_str: str) -> torch.device:
    requested = str(device_str).lower()
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("Requested device 'mps' is unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def _parse_sweep_run_id(run_id: str, run_prefix: str) -> Optional[Dict[str, Any]]:
    pattern = re.compile(
        rf"^{re.escape(run_prefix)}_(?P<dataset>.+?)_fps(?P<fps>[0-9p]+)_seq(?P<seq>[0-9p]+)s_fold(?P<fold>\d+)_"
    )
    match = pattern.match(run_id)
    if not match:
        return None
    return {
        "dataset": match.group("dataset"),
        "fps": _parse_fmt_num(match.group("fps")),
        "seq_len_seconds": _parse_fmt_num(match.group("seq")),
        "fold": int(match.group("fold")),
    }


def _parse_fmt_num(text: str) -> float:
    return float(text.replace("p", "."))


def _parse_imgsz_from_run_id(run_id: str) -> int:
    match = re.search(r"_yolon(?P<imgsz>\d+)_", run_id)
    if not match:
        raise ValueError(f"Could not infer imgsz from run_id: {run_id}")
    return int(match.group("imgsz"))


def _is_fold_complete(data_root: Path, run_id: str, expected_epochs: int) -> bool:
    run_dir = runs_dir(data_root) / run_id
    best_ckpt = run_dir / "checkpoints" / "best.pth"
    metrics_path = run_dir / "metrics.jsonl"
    eval_path = run_dir / "eval.json"
    if eval_path.exists() and best_ckpt.exists():
        return True
    if not best_ckpt.exists() or not metrics_path.exists() or expected_epochs <= 0:
        return False

    records: List[Dict[str, Any]] = []
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except Exception:
        return False

    last_epoch = max(int(record.get("epoch", 0)) for record in records) if records else 0
    if last_epoch >= expected_epochs:
        return True
    return _stopped_by_early_stopping(run_dir, records)


def _stopped_by_early_stopping(run_dir: Path, records: List[Dict[str, Any]]) -> bool:
    config_path = run_dir / "config.json"
    if not config_path.exists() or not records:
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    except Exception:
        return False

    patience = max(0, int(cfg.get("early_stopping_patience", 0)))
    min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    if patience <= 0:
        return False

    best_metric = float("-inf")
    epochs_without_improvement = 0
    for record in sorted(records, key=lambda item: int(item.get("epoch", 0))):
        metric_value = float(record.get("balanced_accuracy", 0.0))
        if metric_value > (best_metric + min_delta):
            best_metric = metric_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            return True
    return False


def _cleanup_completed_fold_dataset(data_root: Path, run_id: str) -> None:
    dataset_dir = datasets_dir(data_root) / run_id
    if not dataset_dir.exists():
        return

    run_dir = runs_dir(data_root) / run_id
    scaler_src = dataset_dir / "scaler.joblib"
    scaler_dst = run_dir / "scaler.joblib"
    try:
        if scaler_src.exists() and not scaler_dst.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(scaler_src, scaler_dst)
        shutil.rmtree(dataset_dir)
        logger.info("Cleaned completed dataset dir: %s", dataset_dir)
    except Exception as exc:
        logger.warning("Failed to clean dataset dir %s: %s", dataset_dir, exc)
