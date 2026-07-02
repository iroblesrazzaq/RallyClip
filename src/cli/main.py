from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from extraction.pose_extractor import PoseExtractor
from features.feature_engineer import FeatureEngineer
from infer import (
    extract_segments_from_binary,
    gaussian_filter1d,
    hysteresis_threshold,
    load_scaler_asset,
    load_model_from_checkpoint,
    run_windowed_inference_average_onnx_stream,
    run_windowed_inference_average_torch_stream,
    write_segments_csv,
)
from preprocessing.data_preprocessor import DataPreprocessor
from runtime.video_validation import VideoValidationError, validate_video
from segmentation.segment import segment_video
from rallyclip_core.contracts import RunRequest, RuntimeDeps, UnsupportedPipelineError
from rallyclip_api import RallyClipServices

try:  # Python 3.11+ ships tomllib; fall back to tomli otherwise.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - dependent on interpreter version
    import tomli as tomllib

from runtime.assets import (  # noqa: E402 — re-exported under legacy names
    YOLO_SIZE_MAP,
    candidate_roots as _candidate_roots,
    manifest_values as _manifest_values,
    resolve_asset as _resolve_asset,
)


@dataclass
class RunConfig:
    # Paths / IO targets
    video_path: Path
    output_dir: Path
    output_name: Optional[str]
    csv_output_dir: Path
    write_csv: bool
    segment_video: bool
    yolo_weights: str
    yolo_device: Optional[str]
    model_path: Path
    scaler_path: Path
    # Immutable contract (resolved from manifest; no phantom defaults)
    fps: float
    seq_len: int
    imgsz: int
    conf: float
    feature_set: str
    screen_width: int
    screen_height: int
    # Mutable postprocess (config -> manifest)
    overlap: int
    sigma: float
    low: float
    high: float
    min_dur_sec: float
    # Runtime IO prefs (safe defaults are fine here)
    start_time: int = 0
    duration: int = 999999
    manifest_path: Optional[Path] = None
    pipeline_id: Optional[str] = None


def _load_config_dict(path: Optional[str]) -> Dict[str, Any]:
    if path:
        cfg_path = Path(path).expanduser()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found at '{cfg_path}'")
    else:
        cfg_path = Path("config.toml")
        if not cfg_path.exists():
            return {}
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _pick_bool(arg_val: Optional[bool], cfg_val: Optional[Any], default: bool) -> bool:
    if arg_val is not None:
        return bool(arg_val)
    if cfg_val is not None:
        return bool(cfg_val)
    return default


# Immutable contract fields: the model's identity. Sourced from the manifest only;
# config.toml has no authority over these. An explicit CLI flag may override (with a
# warning), since that is a deliberate experiment. See docs/runtime-config-refactor-plan.md.
_CONTRACT_FIELDS = (
    "fps", "seq_len", "imgsz", "conf", "feature_set", "screen_width", "screen_height", "yolo_model",
)


def _num_differs(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) > 1e-9
    except (TypeError, ValueError):
        return str(a) != str(b)


def _resolve_contract(name: str, cli_val: Any, manifest_val: Any, cli_flag: Optional[str] = None) -> Any:
    """Immutable field: manifest is authoritative; explicit CLI override warns; crash if unresolved.

    cli_flag is the real argparse flag that overrides this field, if one exists. Some contract
    fields (feature_set, screen_width, screen_height) are manifest-only and have no CLI flag, so
    the error message must not advertise one.
    """
    if cli_val is not None:
        if manifest_val is not None and _num_differs(cli_val, manifest_val):
            logging.warning(
                "Overriding model-contract field '%s'=%r (manifest=%r). This changes the model's "
                "inputs; expect degraded inference unless intentional.", name, cli_val, manifest_val,
            )
        return cli_val
    if manifest_val is not None:
        return manifest_val
    override = f", or pass {cli_flag} to override" if cli_flag else ""
    raise SystemExit(
        f"Required contract field '{name}' is missing from the model manifest and was not passed "
        f"on the CLI. Use a model artifact with a complete manifest.json "
        f"(--artifact-dir / --manifest-path){override}."
    )


def _resolve_mutable(name: str, cli_val: Any, cfg_val: Any, manifest_val: Any) -> Any:
    """Mutable postprocess field: CLI -> config -> manifest -> crash."""
    val = cli_val if cli_val is not None else (cfg_val if cfg_val is not None else manifest_val)
    if val is None:
        raise SystemExit(
            f"Postprocess parameter '{name}' is missing from both config and the model manifest. "
            f"Set it in config.toml or use a manifest with a postprocess section."
        )
    return val


def build_run_config(args: argparse.Namespace) -> RunConfig:
    def arg(key: str, default=None):
        return getattr(args, key, default)

    # config.toml is always loaded (explicit --config, else ./config.toml). It owns mutable
    # postprocess + IO only; it has no authority over immutable contract fields.
    cfg_path = arg("config") or ("config.toml" if Path("config.toml").exists() else None)
    cfg_dict = _load_config_dict(cfg_path) if cfg_path else {}
    cfg_section = cfg_dict.get("run", cfg_dict) if isinstance(cfg_dict, dict) else {}

    def cfg(key: str, default=None):
        return cfg_section.get(key, default) if isinstance(cfg_section, dict) else default

    # A leftover/legacy config.toml must not silently corrupt the contract: warn + ignore.
    for field in _CONTRACT_FIELDS:
        if cfg(field) is not None:
            logging.warning(
                "config.toml sets contract field '%s'; ignoring it (the model manifest is "
                "authoritative). Pass --%s to override intentionally.", field, field.replace("_", "-"),
            )

    video_path = arg("video") or cfg("video_path")
    output_dir = arg("output_dir") or cfg("output_dir") or str(Path.cwd() / "output_videos")
    if not video_path:
        raise SystemExit("Please provide a video via CLI flag or config [run] section.")

    output_name = arg("output_name") or cfg("output_name")
    csv_output_dir_raw = arg("csv_output_dir") or cfg("csv_output_dir")
    yolo_device = arg("yolo_device") or cfg("yolo_device")

    # IO prefs: CLI -> config -> sane default.
    write_csv = _pick_bool(arg("write_csv"), cfg("write_csv"), False)
    segment_video_flag = _pick_bool(arg("segment_video"), cfg("segment_video"), True)

    # Asset resolution. Legacy .pth/.joblib fallbacks removed: every model is a
    # manifest-driven artifact (model.onnx + scaler.json + manifest.json).
    artifact_dir_raw = arg("artifact_dir") or cfg("artifact_dir")
    artifact_dir = Path(artifact_dir_raw).expanduser().resolve() if artifact_dir_raw else None
    model_path_raw = arg("model_path") or cfg("model_path")
    scaler_path_raw = arg("scaler_path") or cfg("scaler_path")
    manifest_path_raw = arg("manifest_path") or cfg("manifest_path")
    if artifact_dir is not None:
        model_path_raw = model_path_raw or str(artifact_dir / "model.onnx")
        scaler_path_raw = scaler_path_raw or str(artifact_dir / "scaler.json")
        manifest_path_raw = manifest_path_raw or str(artifact_dir / "manifest.json")

    model_path = _resolve_asset(
        model_path_raw,
        env_var="RALLYCLIP_MODEL_PATH",
        relatives=["models/rallyclip_v0.3.1/model.onnx"],
        description="RallyClip model artifact (ONNX)",
    )
    scaler_path = _resolve_asset(
        scaler_path_raw,
        env_var="RALLYCLIP_SCALER_PATH",
        relatives=["models/rallyclip_v0.3.1/scaler.json"],
        description="RallyClip scaler artifact (JSON)",
    )
    manifest_path = Path(manifest_path_raw).expanduser().resolve() if manifest_path_raw else None
    manifest = _manifest_values(model_path, manifest_path)

    # Immutable contract: manifest is authoritative; only an explicit CLI flag overrides (warns).
    fps = float(_resolve_contract("fps", arg("fps"), manifest.get("fps"), "--fps"))
    seq_len = int(_resolve_contract("seq_len", arg("seq_len"), manifest.get("seq_len"), "--seq-len"))
    imgsz = int(_resolve_contract("imgsz", arg("imgsz"), manifest.get("imgsz"), "--imgsz"))
    conf = float(_resolve_contract("conf", arg("conf"), manifest.get("conf"), "--conf"))
    feature_set = str(_resolve_contract("feature_set", None, manifest.get("feature_set")))
    screen_width = int(_resolve_contract("screen_width", None, manifest.get("screen_width")))
    screen_height = int(_resolve_contract("screen_height", None, manifest.get("screen_height")))

    cli_yolo = YOLO_SIZE_MAP.get(arg("yolo_size"), arg("yolo_size")) if arg("yolo_size") else None
    yolo_weights = str(_resolve_contract("yolo_model", cli_yolo, manifest.get("yolo_model"), "--yolo-size"))

    # Mutable postprocess: CLI -> config -> manifest -> crash.
    overlap = int(_resolve_mutable("overlap", arg("overlap"), cfg("overlap"), manifest.get("overlap")))
    sigma = float(_resolve_mutable("sigma", arg("sigma"), cfg("sigma"), manifest.get("sigma")))
    low = float(_resolve_mutable("low", arg("low"), cfg("low"), manifest.get("low")))
    high = float(_resolve_mutable("high", arg("high"), cfg("high"), manifest.get("high")))
    min_dur_sec = float(_resolve_mutable("min_dur_sec", arg("min_dur_sec"), cfg("min_dur_sec"), manifest.get("min_dur_sec")))

    return RunConfig(
        video_path=Path(video_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        output_name=output_name,
        csv_output_dir=Path(csv_output_dir_raw).expanduser().resolve() if csv_output_dir_raw else Path(video_path).expanduser().resolve().parent,
        write_csv=write_csv,
        segment_video=segment_video_flag,
        yolo_weights=yolo_weights,
        yolo_device=yolo_device,
        model_path=model_path,
        scaler_path=scaler_path,
        manifest_path=manifest_path,
        pipeline_id=arg("pipeline_id") or cfg("pipeline_id"),
        fps=fps,
        seq_len=seq_len,
        imgsz=imgsz,
        conf=conf,
        feature_set=feature_set,
        screen_width=screen_width,
        screen_height=screen_height,
        overlap=overlap,
        sigma=sigma,
        low=low,
        high=high,
        min_dur_sec=min_dur_sec,
        start_time=int(arg("start_time") if arg("start_time") is not None else cfg("start_time", 0)),
        duration=int(arg("duration") if arg("duration") is not None else cfg("duration", 999999)),
    )


def run_pipeline(cfg: RunConfig) -> int:
    if not cfg.video_path.exists():
        print(f"Error: video file not found at '{cfg.video_path}'")
        return 1

    # Preflight the input before the expensive pose/preprocess passes.
    try:
        validate_video(cfg.video_path, seq_len=int(cfg.seq_len), fps=float(cfg.fps))
    except VideoValidationError as exc:
        print(f"Error: {exc}")
        return 1

    if cfg.yolo_device:
        os.environ["POSE_DEVICE"] = cfg.yolo_device
    request = RunRequest(
        video_path=cfg.video_path,
        output_dir=cfg.output_dir,
        output_name=cfg.output_name,
        csv_output_dir=cfg.csv_output_dir,
        write_csv=cfg.write_csv,
        segment_video=cfg.segment_video,
        yolo_weights=cfg.yolo_weights,
        yolo_device=cfg.yolo_device,
        model_path=cfg.model_path,
        scaler_path=cfg.scaler_path,
        manifest_path=cfg.manifest_path,
        pipeline_id=cfg.pipeline_id,
        fps=cfg.fps,
        seq_len=cfg.seq_len,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        feature_set=cfg.feature_set,
        screen_width=cfg.screen_width,
        screen_height=cfg.screen_height,
        overlap=cfg.overlap,
        sigma=cfg.sigma,
        low=cfg.low,
        high=cfg.high,
        min_dur_sec=cfg.min_dur_sec,
        start_time=cfg.start_time,
        duration=cfg.duration,
        models_dir=Path.cwd() / "models",
    )
    deps = RuntimeDeps(
        np=np,
        PoseExtractor=PoseExtractor,
        DataPreprocessor=DataPreprocessor,
        FeatureEngineer=FeatureEngineer,
        load_scaler_asset=load_scaler_asset,
        load_model_from_checkpoint=load_model_from_checkpoint,
        run_windowed_inference_average_onnx_stream=run_windowed_inference_average_onnx_stream,
        run_windowed_inference_average_torch_stream=run_windowed_inference_average_torch_stream,
        gaussian_filter1d=gaussian_filter1d,
        hysteresis_threshold=hysteresis_threshold,
        extract_segments_from_binary=extract_segments_from_binary,
        write_segments_csv=write_segments_csv,
        segment_video=segment_video,
        apply_pose_device=None,
    )
    try:
        RallyClipServices().run_analysis(request, deps=deps)
    except UnsupportedPipelineError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"✅ Done. Outputs in {cfg.output_dir}")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1].lower() == "gui":
        sys.argv.pop(1)
        try:
            from gui.app import launch
        except SystemExit:
            raise
        except Exception as exc:  # pragma: no cover - runtime safety
            print("rallyclip gui requires Flask. Reinstall with `pip install .`.", file=sys.stderr)
            print(f"Details: {exc}", file=sys.stderr)
            return 1
        return launch()

    p = argparse.ArgumentParser(description="RallyClip end-to-end CLI with optional config.toml.")
    p.add_argument("--config", help="Path to config.toml. If omitted, looks for ./config.toml.")
    p.add_argument("--video", required=False, help="Path to input MP4 video (required unless set in config)")
    p.add_argument("--output-dir", help="Directory to store outputs (defaults to ./output_videos)")
    p.add_argument("--output-name", help="Optional base name for outputs (without extension)")
    p.add_argument("--csv-output-dir", help="Optional directory for CSV output (defaults to video directory)")
    p.add_argument("--model-path", help="Path to model artifact (.onnx preferred, legacy .pth supported)")
    p.add_argument("--scaler-path", help="Path to scaler artifact (.json preferred, legacy .joblib supported)")
    p.add_argument("--artifact-dir", help="Directory containing model.onnx, scaler.json, and manifest.json")
    p.add_argument("--manifest-path", help="Path to model manifest.json for runtime defaults")
    p.add_argument("--pipeline-id", help="Explicit analysis pipeline override for compatible artifacts")
    p.add_argument("--yolo-size", choices=list(YOLO_SIZE_MAP.keys()), help="YOLO pose model size (auto-downloads if needed)")
    p.add_argument("--yolo-device", choices=["cpu", "cuda", "mps"], help="Force YOLO device (overrides POSE_DEVICE env)")

    p.add_argument("--fps", type=float, help="Sampling FPS used during feature creation")
    p.add_argument("--seq-len", type=int, help="Sequence length for inference windows")
    p.add_argument("--overlap", type=int, help="Overlap (frames) between windows")
    p.add_argument("--sigma", type=float, help="Gaussian smoothing sigma")
    p.add_argument("--low", type=float, help="Hysteresis low threshold")
    p.add_argument("--high", type=float, help="Hysteresis high threshold")
    p.add_argument("--min-dur-sec", type=float, help="Minimum segment duration in seconds")
    p.add_argument("--conf", type=float, help="Pose model confidence threshold")
    p.add_argument("--imgsz", type=int, help="YOLO inference image size")
    p.add_argument("--start-time", type=int, help="Start time offset (seconds)")
    p.add_argument("--duration", type=int, help="Max duration to process (seconds)")

    p.add_argument("--write-csv", dest="write_csv", action="store_true", default=None, help="Write segments CSV")
    p.add_argument("--no-csv", dest="write_csv", action="store_false", help="Skip writing segments CSV")
    p.add_argument("--segment-video", dest="segment_video", action="store_true", default=None, help="Write segmented MP4")
    p.add_argument("--no-segment-video", dest="segment_video", action="store_false", help="Skip segmented MP4")
    args = p.parse_args()

    cfg = build_run_config(args)
    return run_pipeline(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
