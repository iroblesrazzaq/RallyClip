from __future__ import annotations

from copy import deepcopy
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    pre_steps = [str(s) for s in sweep_cfg.get("steps_precompute", ["preprocess", "features"])]
    fold_steps = [str(s) for s in sweep_cfg.get("steps_per_fold", ["dataset", "train", "eval"])]
    resume = bool(sweep_cfg.get("resume", True))
    run_prefix = str(sweep_cfg.get("run_prefix", "cv"))
    val_ratio = float(loso_cfg.get("val_ratio", config.get("dataset", {}).get("split", {}).get("val_ratio", 0.1)))
    seed = int(loso_cfg.get("seed", config.get("dataset", {}).get("split", {}).get("seed", 1337)))

    logger.info(
        "Sweep start: datasets=%d fps=%s seq_len=%s folds=%d",
        len(dataset_specs),
        fps_values,
        seq_values,
        len(videos),
    )

    for ds in dataset_specs:
        ds_name = str(ds["name"])
        for fps in fps_values:
            pre_cfg = deepcopy(config)
            _apply_sweep_dataset(pre_cfg, ds)
            pre_cfg.setdefault("preprocess", {})["target_fps"] = float(fps)
            pre_cfg["run_id"] = f"{run_prefix}_{ds_name}_fps{_fmt_num(fps)}_prep"
            run_pipeline(pre_cfg, steps_override=pre_steps)

            for seq_len in seq_values:
                for fold_idx, held_out_video in enumerate(videos, start=1):
                    fold_cfg = deepcopy(config)
                    _apply_sweep_dataset(fold_cfg, ds)
                    fold_cfg.setdefault("preprocess", {})["target_fps"] = float(fps)
                    fold_cfg.setdefault("dataset", {})["seq_len_seconds"] = float(seq_len)
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
                    if resume and best_ckpt.exists():
                        logger.info("Skipping completed fold: %s", run_id)
                        continue

                    run_pipeline(fold_cfg, steps_override=fold_steps)

    logger.info("Sweep complete")


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
    builder.build(feature_root, output_dir, videos, feature_set)


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
    return {"name": _slug(name, 48), "yolo": yolo_base, "extract": dict(entry.get("extract", {}))}


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
