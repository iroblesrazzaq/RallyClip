from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

from rallyclip_core.contracts import (
    CancelCheck,
    FrameSegment,
    PipelineSpec,
    ProgressCallback,
    ProgressEvent,
    RunRequest,
    RunResult,
    RuntimeDeps,
    UnsupportedPipelineError,
)
from rallyclip_core.intervals import frame_segments_to_intervals
from rallyclip_core.pipelines import FRAME_PROBABILITY_HYSTERESIS, START_END_ATTENTION_VOTING


def _emit(
    callback: Optional[ProgressCallback],
    stage: str,
    progress: int,
    status: str = "in_progress",
    metadata: Optional[dict] = None,
) -> None:
    if callback is not None:
        callback(ProgressEvent(stage=stage, progress=progress, status=status, metadata=metadata or {}))


def _check(cancel_check: Optional[CancelCheck]) -> None:
    if cancel_check is not None:
        cancel_check()


def _estimate_stream_window_count(num_frames: int, sequence_length: int, overlap: int) -> Optional[int]:
    if num_frames < sequence_length or sequence_length <= 0 or overlap < 0 or overlap >= sequence_length:
        return None
    step = sequence_length - overlap
    count = 1 + max(0, (num_frames - sequence_length) // step)
    last_start = (count - 1) * step
    if last_start + sequence_length < num_frames:
        count += 1
    return count


class AnalysisModel(ABC):
    """Artifact-bound analysis model.

    A model owns the full path from source video to point intervals: preprocessing,
    inference, and postprocessing. Different future artifacts can provide different
    implementations behind this same output contract.
    """

    pipeline_id: str

    def __init__(self, request: RunRequest, spec: PipelineSpec, deps: RuntimeDeps) -> None:
        self.request = request
        self.spec = spec
        self.deps = deps

    @abstractmethod
    def preprocess(self, progress_callback: Optional[ProgressCallback], cancel_check: Optional[CancelCheck]):
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        features,
        progress_callback: Optional[ProgressCallback],
        cancel_check: Optional[CancelCheck],
    ):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, model_output) -> List[FrameSegment]:
        raise NotImplementedError

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> RunResult:
        features = self.preprocess(progress_callback, cancel_check)
        model_output = self.infer(features, progress_callback, cancel_check)
        frame_segments = self.postprocess(model_output)
        _emit(progress_callback, "inference", 100, "completed")
        return self._write_outputs(frame_segments)

    def _write_outputs(self, frame_segments: List[FrameSegment]) -> RunResult:
        request = self.request
        intervals_sec = frame_segments_to_intervals(frame_segments, float(request.fps))
        csv_path = None
        video_path = None
        base_name = request.output_name or request.video_path.stem
        if request.write_csv:
            request.csv_output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = request.csv_output_dir / f"{base_name}_segments.csv"
            self.deps.write_segments_csv(frame_segments, str(csv_path), fps=float(request.fps), overwrite=True)
        if request.segment_video:
            request.output_dir.mkdir(parents=True, exist_ok=True)
            video_path = request.output_dir / f"{base_name}_segmented.mp4"
            if intervals_sec:
                self.deps.segment_video(str(request.video_path), intervals_sec, str(video_path))
        return RunResult(
            frame_segments=frame_segments,
            intervals_sec=intervals_sec,
            csv_path=csv_path,
            video_path=video_path,
            diagnostics={"pipeline_id": self.pipeline_id},
        )


class FrameProbabilityHysteresisModel(AnalysisModel):
    pipeline_id = FRAME_PROBABILITY_HYSTERESIS

    def preprocess(self, progress_callback: Optional[ProgressCallback], cancel_check: Optional[CancelCheck]):
        request = self.request
        deps = self.deps
        if str(request.feature_set) != "v1":
            raise UnsupportedPipelineError(
                f"This model declares feature_set='{request.feature_set}', but only 'v1' is implemented "
                "for pipeline 'frame_probability_hysteresis'."
            )

        yolo_weights = request.yolo_weights
        if not Path(yolo_weights).is_absolute():
            # Resolve bare weight names (from the manifest) against the models
            # directory and the manifest's own directory, so every consumer —
            # court detection included — gets a concrete file path. Names that
            # don't resolve stay bare and fall through to ultralytics' own
            # download/resolution, as before.
            search_dirs = [Path(request.models_dir or (Path.cwd() / "models"))]
            if request.manifest_path is not None:
                search_dirs.append(Path(request.manifest_path).parent)
            for base in search_dirs:
                candidate = base / yolo_weights
                if candidate.exists():
                    yolo_weights = str(candidate)
                    break
        pose_device = None
        if request.yolo_device and deps.apply_pose_device is not None:
            pose_device = deps.apply_pose_device(str(request.yolo_device), model_path=yolo_weights, set_env=False)
        elif deps.apply_pose_device is not None:
            pose_device = deps.apply_pose_device(None, model_path=yolo_weights, set_env=False)

        pre = deps.DataPreprocessor(
            screen_width=int(request.screen_width),
            screen_height=int(request.screen_height),
            save_court_masks=False,
            yolo_model_path=yolo_weights,
            conf=float(request.conf),
            **({"yolo_device": pose_device} if pose_device is not None else {}),
        )
        _check(cancel_check)
        _emit(progress_callback, "pose", 1)
        court_mask, _ = pre.compute_court_mask(str(request.video_path))
        _emit(progress_callback, "pose", 3)

        _check(cancel_check)
        extractor_kwargs = {
            "model_path": yolo_weights,
            "model_dir": str(request.models_dir or (Path.cwd() / "models")),
        }
        if pose_device is not None:
            extractor_kwargs["device"] = pose_device
        extractor = deps.PoseExtractor(**extractor_kwargs)

        def pose_progress(frac: float, meta=None) -> None:
            _check(cancel_check)
            # meta (frames_seen/frames_total/smoothed fps) feeds client-side
            # ETA display; forward it rather than dropping it here.
            _emit(progress_callback, "pose", int(3 + max(0.0, min(1.0, frac)) * 96), metadata=meta or {})

        src_height, src_width, _ = pre._source_frame_shape(str(request.video_path))
        pose_stream = extractor.iter_pose_frames(
            video_path=str(request.video_path),
            confidence_threshold=float(request.conf),
            start_time_seconds=int(request.start_time),
            duration_seconds=int(request.duration),
            target_fps=int(request.fps),
            imgsz=int(request.imgsz),
            annotations_csv=None,
            progress_callback=pose_progress,
        )
        preprocessed_stream = pre.iter_preprocess_frames(pose_stream, court_mask, src_width, src_height)
        fe = deps.FeatureEngineer(
            screen_width=int(request.screen_width),
            screen_height=int(request.screen_height),
            target_fps=float(request.fps),
        )
        feature_stream = fe.iter_build_features(preprocessed_stream)
        _emit(progress_callback, "preprocess", 1)
        _emit(progress_callback, "feature", 1)
        return feature_stream

    def infer(
        self,
        features,
        progress_callback: Optional[ProgressCallback],
        cancel_check: Optional[CancelCheck],
    ):
        request = self.request
        deps = self.deps
        _emit(progress_callback, "inference", 5)
        scaler = deps.load_scaler_asset(str(request.scaler_path))

        duration_s = request.estimated_duration_s
        if duration_s is None:
            duration_s = float(request.duration) if 0 < int(request.duration) < 999999 else 0.0
        estimated_feature_rows = max(0, int(round(float(duration_s) * float(request.fps))) - 1)
        estimated_windows = _estimate_stream_window_count(
            estimated_feature_rows,
            int(request.seq_len),
            int(request.overlap),
        )
        feature_rows_seen = 0

        def scaled_feature_rows():
            nonlocal feature_rows_seen
            for feature_vector, _target in features:
                _check(cancel_check)
                feature_rows_seen += 1
                if estimated_feature_rows > 0:
                    frac = min(1.0, feature_rows_seen / float(estimated_feature_rows))
                    stage_progress = int(1 + frac * 94)
                    _emit(progress_callback, "preprocess", stage_progress)
                    _emit(progress_callback, "feature", stage_progress)
                    if estimated_windows is None:
                        _emit(progress_callback, "inference", int(5 + frac * 80))
                row = deps.np.asarray(feature_vector, dtype=deps.np.float32).reshape(1, -1)
                yield scaler.transform(row)[0].astype(deps.np.float32)

        def infer_progress(frac: float) -> None:
            _emit(progress_callback, "inference", int(5 + max(0.0, min(1.0, frac)) * 90))

        if request.model_path.suffix.lower() == ".onnx":
            output = deps.run_windowed_inference_average_onnx_stream(
                str(request.model_path),
                scaled_feature_rows(),
                sequence_length=int(request.seq_len),
                overlap=int(request.overlap),
                progress_callback=infer_progress,
                total_windows=estimated_windows,
            )
        else:
            model, device = deps.load_model_from_checkpoint(str(request.model_path), return_logits=False)
            output = deps.run_windowed_inference_average_torch_stream(
                model,
                device,
                scaled_feature_rows(),
                sequence_length=int(request.seq_len),
                overlap=int(request.overlap),
                progress_callback=infer_progress,
                total_windows=estimated_windows,
            )

        _emit(progress_callback, "pose", 100, "completed")
        _emit(progress_callback, "preprocess", 100, "completed")
        _emit(progress_callback, "feature", 100, "completed")
        return output

    def postprocess(self, model_output) -> List[FrameSegment]:
        deps = self.deps
        request = self.request
        smoothed_probs = deps.gaussian_filter1d(model_output.astype(deps.np.float32), sigma=float(request.sigma))
        min_duration_frames = int(round(max(0.0, float(request.min_dur_sec)) * float(request.fps)))
        binary_pred = deps.hysteresis_threshold(
            smoothed_probs,
            low=float(request.low),
            high=float(request.high),
            min_duration=min_duration_frames,
        )
        return list(deps.extract_segments_from_binary(binary_pred))


class StartEndAttentionVotingModel(AnalysisModel):
    pipeline_id = START_END_ATTENTION_VOTING

    def preprocess(self, progress_callback: Optional[ProgressCallback], cancel_check: Optional[CancelCheck]):
        raise UnsupportedPipelineError(
            "Pipeline 'start_end_attention_voting' is declared but no production runtime is implemented yet."
        )

    def infer(self, features, progress_callback: Optional[ProgressCallback], cancel_check: Optional[CancelCheck]):
        raise UnsupportedPipelineError(
            "Pipeline 'start_end_attention_voting' is declared but no production runtime is implemented yet."
        )

    def postprocess(self, model_output) -> List[FrameSegment]:
        return decode_start_end_votes(
            model_output["start_scores"],
            model_output["end_scores"],
            threshold=float(model_output.get("threshold", 0.5)),
            min_duration_frames=int(model_output.get("min_duration_frames", 1)),
        )


def decode_start_end_votes(
    start_scores,
    end_scores,
    *,
    threshold: float = 0.5,
    min_duration_frames: int = 1,
) -> List[FrameSegment]:
    """Pair start/end score peaks into intervals.

    This is intentionally small and deterministic for now; the real E2E model can
    replace the voting rule behind the same postprocess contract.
    """
    starts = [idx for idx, value in enumerate(start_scores) if float(value) >= threshold]
    ends = [idx for idx, value in enumerate(end_scores) if float(value) >= threshold]
    intervals: List[FrameSegment] = []
    end_cursor = 0
    for start in starts:
        while end_cursor < len(ends) and ends[end_cursor] <= start:
            end_cursor += 1
        if end_cursor >= len(ends):
            break
        end = ends[end_cursor]
        if end - start >= max(1, min_duration_frames):
            intervals.append((start, end))
        end_cursor += 1
    return intervals


def build_analysis_model(request: RunRequest, spec: PipelineSpec, deps: RuntimeDeps) -> AnalysisModel:
    if spec.pipeline_id == FRAME_PROBABILITY_HYSTERESIS:
        return FrameProbabilityHysteresisModel(request, spec, deps)
    if spec.pipeline_id == START_END_ATTENTION_VOTING:
        return StartEndAttentionVotingModel(request, spec, deps)
    raise UnsupportedPipelineError(f"Unsupported pipeline_id '{spec.pipeline_id}'.")

