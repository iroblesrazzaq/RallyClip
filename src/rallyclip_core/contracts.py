from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


Interval = Tuple[float, float]
FrameSegment = Tuple[int, int]


class RallyClipError(RuntimeError):
    """Base class for clean, user-facing runtime errors."""


class UnsupportedPipelineError(RallyClipError):
    """Raised when a requested pipeline is unknown or incompatible."""


@dataclass(frozen=True)
class RunRequest:
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
    fps: float
    seq_len: int
    imgsz: int
    conf: float
    feature_set: str
    screen_width: int
    screen_height: int
    overlap: int
    sigma: float
    low: float
    high: float
    min_dur_sec: float
    start_time: int = 0
    duration: int = 999999
    pipeline_id: Optional[str] = None
    manifest_path: Optional[Path] = None
    models_dir: Optional[Path] = None
    estimated_duration_s: Optional[float] = None


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    feature_set: str
    model_output: str
    decode_method: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    progress: int
    status: str = "in_progress"
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    frame_segments: List[FrameSegment]
    intervals_sec: List[Interval]
    csv_path: Optional[Path] = None
    video_path: Optional[Path] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]
CancelCheck = Callable[[], None]


@dataclass(frozen=True)
class SavedMatch:
    id: str
    title: str
    source_path: Path
    csv_path: Path
    thumbnail_path: Optional[Path]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlaybackManifest:
    source_duration_s: float
    chunk_duration_s: float
    point_intervals: List[Interval]


@dataclass
class RuntimeDeps:
    """Runtime dependencies for the engine.

    Tests and GUI code can inject fakes here; production builds use the lazy
    loader in rallyclip_engine.runtime.
    """

    np: Any
    PoseExtractor: Any
    DataPreprocessor: Any
    FeatureEngineer: Any
    load_scaler_asset: Callable[..., Any]
    load_model_from_checkpoint: Callable[..., Any]
    run_windowed_inference_average_onnx_stream: Callable[..., Any]
    run_windowed_inference_average_torch_stream: Callable[..., Any]
    gaussian_filter1d: Callable[..., Any]
    hysteresis_threshold: Callable[..., Any]
    extract_segments_from_binary: Callable[..., Any]
    write_segments_csv: Callable[..., Any]
    segment_video: Callable[..., Any]
    apply_pose_device: Optional[Callable[..., Any]] = None
    # Heatmap pipeline (optional: engine falls back to lazy `infer` imports when
    # unset, so pre-existing RuntimeDeps constructions keep working unchanged).
    run_multitrack_windowed_inference_onnx_stream: Optional[Callable[..., Any]] = None
    decode_heatmap_segments: Optional[Callable[..., Any]] = None
    heatmap_decode_config_cls: Optional[Any] = None
