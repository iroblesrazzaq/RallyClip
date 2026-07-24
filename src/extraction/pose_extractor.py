import os
import time
from typing import Callable, Optional

import av
import logging
import numpy as np
from tqdm import tqdm


def _pipeline_profile() -> str:
    profile = os.environ.get("PIPELINE_PROFILE", "").strip().lower()
    return profile or "main"



class PoseExtractionCancelled(Exception):
    """Raised when upstream caller requests pose extraction cancellation."""


class PoseExtractor:
    """
    YOLOv8-pose based extractor that converts input video frames into pose arrays
    and saves compressed npz artifacts. Designed for reuse by CLI and GUI.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_path: str = "yolov8s-pose.pt",
        # Fallback only: production callers pass imgsz per call from the model
        # manifest (v0.3.1 contract: 960). Keep this aligned with the shipped
        # contract so a missing per-call value cannot silently change scale.
        imgsz: int = 960,
        device: Optional[str] = None,
    ) -> None:
        # When device is set, use it as-is. Auto-selected devices should be
        # resolved via runtime.device.apply_pose_device before construction.
        # Set early so downstream checks can use it
        self.model_path = model_path
        self.model_dir = model_dir
        self.imgsz = int(imgsz)

        profile = _pipeline_profile()

        from runtime.device import prefer_cpu_over_mps_for_pose, resolve_pose_device

        if device is not None:
            self.device = device
        else:
            env_device = os.environ.get("POSE_DEVICE", "").strip().lower()
            device_from_env = env_device in {"cpu", "cuda", "mps", "coreml"}
            self.device = resolve_pose_device(env_device if device_from_env else None)
            if not device_from_env:
                self.device = prefer_cpu_over_mps_for_pose(self.device, self.model_path)
                os.environ["POSE_DEVICE"] = self.device

        env_bs = os.environ.get("POSE_BATCH_SIZE", "").strip()
        batch_from_env = env_bs.isdigit()
        if batch_from_env:
            self.batch_size = int(env_bs)
        else:
            self.batch_size = self._default_batch_size(self.device, profile)

        # Prefer local file if model_dir provided and file exists; otherwise let
        # Ultralytics handle download from model name (e.g., "yolov8s-pose.pt").
        yolo_arg = self.model_path
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            candidate = os.path.join(model_dir, self.model_path)
            if os.path.exists(candidate):
                yolo_arg = candidate

        requested_device = self.device
        if str(yolo_arg).lower().endswith(".onnx"):
            # Torch-free path: onnxruntime runner with the same predict surface.
            from extraction.yolo_onnx_runner import YOLO as OnnxYOLO

            model_arg, providers = self._resolve_onnx_session(yolo_arg)
            try:
                self.model = OnnxYOLO(model_arg, providers=providers)
            except Exception as exc:
                if providers is None:
                    raise  # the CPU reference path failing is a real error
                # Accelerated EP session/compile failures (CoreML OS quirks,
                # CUDA driver/toolkit mismatches) must degrade to CPU, not kill
                # the analysis.
                logging.warning(
                    "POSE: %s session failed (%s); using cpu.",
                    self.device,
                    exc,
                )
                self.device = "cpu"
                self.model = OnnxYOLO(yolo_arg)
        else:
            if self.device == "coreml":
                # CoreML rides the ONNX runner; ultralytics .pt weights can't use it.
                logging.warning("POSE: coreml requires ONNX weights; using cpu for %s.", yolo_arg)
                self.device = "cpu"
            elif self.device == "cuda":
                from runtime.device import torch_cuda_available

                # ORT CUDA must not drive Ultralytics — needs a real torch CUDA build.
                if not torch_cuda_available():
                    logging.warning(
                        "POSE: cuda requested for .pt weights but torch CUDA is "
                        "unavailable; using cpu for %s.",
                        yolo_arg,
                    )
                    self.device = "cpu"
            from ultralytics import YOLO
            from ultralytics.utils import SETTINGS

            if model_dir:
                # Direct ultralytics downloads into the provided models directory
                try:
                    SETTINGS["weights_dir"] = os.path.abspath(model_dir)
                except Exception:
                    pass
            self.model = YOLO(yolo_arg)
            try:
                self.model.to(self.device)
            except Exception as exc:
                if self.device != "cpu":
                    logging.warning(
                        "POSE: model.to(%s) failed (%s); using cpu.",
                        self.device,
                        exc,
                    )
                    self.device = "cpu"
                    try:
                        self.model.to("cpu")
                    except Exception:
                        pass
        if self.device != requested_device and not batch_from_env:
            # CUDA (batch=8) / other accelerated picks must not keep a large
            # batch after falling back to the CPU path (per-frame streaming).
            self.batch_size = self._default_batch_size(self.device, profile)
        if (
            requested_device in {"coreml", "cuda"}
            and self.device != requested_device
            and os.environ.get("POSE_DEVICE", "").strip().lower() == requested_device
        ):
            # Accelerated request degraded to CPU: sync the env so later
            # extractors in this process don't retry (and re-warn about) the
            # same unavailable path.
            os.environ["POSE_DEVICE"] = self.device

    @staticmethod
    def _default_batch_size(device: str, profile: str) -> int:
        if profile == "mvp":
            return 1 if device == "cpu" else 16
        if device == "mps":
            return 2
        if device == "cuda":
            return 8
        return 1

    def _resolve_onnx_session(self, dynamic_path: str):
        """Pick the ONNX file + ORT providers for the requested device.

        Default (cpu): the dynamic-axes export on the CPU EP — the byte-parity
        reference path. Opt-in device "coreml": the static-shape sibling export
        (*-static.onnx next to the dynamic model) on the CoreML EP, ~8x pose
        throughput on Apple silicon (docs/perf/coreml-spike/) — dynamic axes
        block the Neural Engine, hence the separate static file. Opt-in "cuda":
        the same dynamic export on CUDAExecutionProvider when onnxruntime-gpu
        is installed. Any missing piece degrades to the CPU path with a warning
        instead of failing.
        """
        if self.device == "coreml":
            from runtime.device import coreml_pose_available

            static_path = self._static_onnx_sibling(dynamic_path)
            if not coreml_pose_available():
                logging.warning("POSE: coreml requested but the CoreML EP is unavailable; using cpu.")
            elif static_path is None:
                logging.warning(
                    "POSE: coreml requested but no *-static.onnx export found next to %s; using cpu.",
                    dynamic_path,
                )
            else:
                return static_path, [
                    ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                    "CPUExecutionProvider",
                ]
        elif self.device == "cuda":
            from runtime.device import cuda_pose_available

            if cuda_pose_available():
                return dynamic_path, ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logging.warning(
                "POSE: cuda requested but CUDAExecutionProvider is unavailable; "
                "install the [gpu] extra (onnxruntime-gpu) and ensure the CUDA "
                "toolkit matches. Using cpu."
            )
        self.device = "cpu"
        return dynamic_path, None

    @staticmethod
    def _static_onnx_sibling(dynamic_path: str) -> Optional[str]:
        import glob

        if str(dynamic_path).lower().endswith("-static.onnx"):
            return str(dynamic_path)
        parent = os.path.dirname(str(dynamic_path))
        if not parent:
            # Bare filename: no directory to search — globbing would scan the
            # CWD and could pick up an unrelated *-static.onnx.
            return None
        matches = sorted(glob.glob(os.path.join(parent, "*-static.onnx")))
        if len(matches) <= 1:
            return matches[0] if matches else None
        # Multiple static exports: only proceed on an unambiguous match with
        # the dynamic model's name (its stem minus the "-dynamic"/resolution
        # tokens), never a silent alphabetical pick of the wrong resolution.
        tokens = os.path.basename(str(dynamic_path))[: -len(".onnx")].split("-")
        while tokens and (tokens[-1] == "dynamic" or tokens[-1].isdigit()):
            tokens.pop()
        prefix = "-".join(tokens)
        preferred = [m for m in matches if prefix and os.path.basename(m).startswith(prefix)]
        if len(preferred) == 1:
            return preferred[0]
        logging.warning(
            "POSE: multiple *-static.onnx exports next to %s and no unambiguous match; using cpu.",
            dynamic_path,
        )
        return None

    def frame_iterator_pyav(self, video_path: str):
        # Yields the decoded ``av.VideoFrame`` (NOT a BGR ndarray) so the caller can defer the
        # costly to_ndarray(bgr24) conversion to only the frames it actually keeps. At
        # target_fps << source_fps most frames are downsampled out, so converting eagerly here
        # wasted ~11/12 of the conversions on a 60fps source. See _to_bgr / iter_pose_frames.
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                # Enable FFmpeg's internal multithreaded decode (deterministic, identical
                # pixels) so HEVC/H.264 decode overlaps across threads instead of running
                # single-threaded. Decode is a real slice of wall time on long 1080p clips.
                try:
                    stream.thread_type = "AUTO"
                except Exception:
                    pass
                time_base = stream.time_base
                for frame in container.decode(stream):
                    ts = float(frame.pts * time_base) if frame.pts is not None else None
                    if ts is None:
                        continue
                    yield frame, ts
        except Exception as e:
            logging.error("[PyAV Error] %s", e)
            return

    @staticmethod
    def _to_bgr(frame):
        """Convert a decoded frame to a BGR ndarray, only when a frame is actually kept.

        Handles both real ``av.VideoFrame`` objects (-> to_ndarray) and bare ndarrays (the test
        / bench stubs already yield ndarrays), so the downsample path is identical to before for
        the frames that pass the timestamp gate.
        """
        to_ndarray = getattr(frame, "to_ndarray", None)
        if to_ndarray is not None:
            return to_ndarray(format="bgr24")
        return frame

    def iter_pose_frames(
        self,
        video_path: str,
        confidence_threshold: float,
        start_time_seconds: int = 0,
        duration_seconds: int = 60,
        target_fps: int = 15,
        imgsz: Optional[int] = None,
        annotations_csv: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ):
        """Streaming pose extraction core (generator).

        Yields per-frame data dicts in stream order -- the same sequence
        :meth:`extract_pose_frames` returns as a list -- but only buffers one detection
        batch (plus the downsampled-out frames interleaved with it) at a time, so peak
        memory is O(batch) instead of O(num_frames). Detection results arrive per batch,
        so the buffered chunk is filled then flushed out in order at each batch boundary.
        """
        import csv
        predict_imgsz = int(self.imgsz if imgsz is None else imgsz)

        annotations = None
        if annotations_csv and annotations_csv != "None" and os.path.exists(annotations_csv):
            try:
                with open(annotations_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    reader.fieldnames = [c.strip().lower() for c in reader.fieldnames]
                    starts_list, ends_list = [], []
                    for row in reader:
                        try:
                            starts_list.append(float(row['start_time']))
                            ends_list.append(float(row['end_time']))
                        except (ValueError, KeyError, TypeError):
                            continue
                    if starts_list:
                        annotations = {'starts': np.array(starts_list), 'ends': np.array(ends_list)}
            except Exception as e:
                print(f"Warning: could not read annotations {annotations_csv}: {e}")

        try:
            with av.open(video_path) as container:
                total_frames = container.streams.video[0].frames
        except Exception:
            total_frames = 0

        processed_frames_count = 0  # frames actually processed/saved (downsampled)
        frames_seen = 0  # all frames iterated from the stream
        next_target_timestamp = start_time_seconds
        first_processed_ts = None
        last_report_time = time.time()
        smoothed_proc_fps = 0.0
        smoothed_seen_fps = 0.0
        alpha = 0.9  # smoothing factor for FPS EMA

        EPS = 1e-6
        if annotations is not None:
            starts = annotations['starts']
            ends = annotations['ends']
            annotation_index = 0
            num_annotations = starts.size
        else:
            starts = ends = None
            annotation_index = 0
            num_annotations = 0

        # `pending` holds the frame dicts emitted since the last batch flush, in stream
        # order; processed frames await detection results that the next flush fills in. It
        # is never rebound (only mutated in place + .clear()) so the _flush_batch closure
        # always sees the same list object.
        pending = []
        batch_frames = []
        batch_positions = []  # index within `pending` of each batched (processed) frame

        def _flush_batch():
            nonlocal batch_frames, batch_positions
            if not batch_frames:
                return
            try:
                results = self.model.predict(
                    source=batch_frames,
                    verbose=False,
                    device=self.device,
                    conf=confidence_threshold,
                    imgsz=predict_imgsz,
                    batch=self.batch_size,
                )
            except TypeError:
                results = self.model.predict(
                    source=batch_frames,
                    verbose=False,
                    device=self.device,
                    conf=confidence_threshold,
                    imgsz=predict_imgsz,
                )
            for i, res in enumerate(results):
                pos = batch_positions[i]
                frame_data = {}
                if res is not None and getattr(res, "boxes", None) is not None:
                    try:
                        frame_data["boxes"] = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                        frame_data["box_conf"] = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
                    except Exception:
                        frame_data["boxes"] = np.empty((0, 4), dtype=np.float32)
                        frame_data["box_conf"] = np.empty((0,), dtype=np.float32)
                    try:
                        frame_data["keypoints"] = res.keypoints.xy.detach().cpu().numpy().astype(np.float32)
                        frame_data["conf"] = res.keypoints.conf.detach().cpu().numpy().astype(np.float32)
                    except Exception:
                        frame_data["keypoints"] = np.empty((0, 17, 2), dtype=np.float32)
                        frame_data["conf"] = np.empty((0, 17), dtype=np.float32)
                else:
                    frame_data["boxes"] = np.empty((0, 4), dtype=np.float32)
                    frame_data["box_conf"] = np.empty((0,), dtype=np.float32)
                    frame_data["keypoints"] = np.empty((0, 17, 2), dtype=np.float32)
                    frame_data["conf"] = np.empty((0, 17), dtype=np.float32)
                frame_data["annotation_status"] = pending[pos].get("annotation_status", 0)
                pending[pos] = frame_data
            batch_frames = []
            batch_positions = []

        gen = self.frame_iterator_pyav(video_path)
        show_tqdm = os.environ.get("RALLYCLIP_NO_TQDM", "").strip().lower() not in {"1", "true", "yes"}
        iterator = tqdm(gen, total=total_frames, desc="Processing frames", unit="frame") if show_tqdm else gen
        # simple progress proxy if tqdm not desired in GUI
        for frame, current_timestamp in iterator:
            frames_seen += 1
            appended = False
            annotation_status_current = -1
            if current_timestamp < start_time_seconds:
                continue
            if current_timestamp > (start_time_seconds + duration_seconds):
                break
            if current_timestamp >= next_target_timestamp:
                processed_frames_count += 1
                annotation_status_current = 0
                if num_annotations > 0:
                    while (annotation_index < num_annotations - 1) and (current_timestamp > ends[annotation_index] + EPS):
                        annotation_index += 1
                    if (starts[annotation_index] - EPS) <= current_timestamp <= (ends[annotation_index] + EPS):
                        annotation_status_current = 1
                pending.append({
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "box_conf": np.empty((0,), dtype=np.float32),
                    "keypoints": np.empty((0, 17, 2), dtype=np.float32),
                    "conf": np.empty((0, 17), dtype=np.float32),
                    "annotation_status": annotation_status_current,
                })
                # Convert to BGR only now that this frame passed the downsample gate.
                batch_frames.append(self._to_bgr(frame))
                batch_positions.append(len(pending) - 1)
                appended = True
                if len(batch_frames) >= self.batch_size:
                    _flush_batch()
                    yield from pending
                    pending.clear()
                next_target_timestamp += (1.0 / target_fps)
                if first_processed_ts is None:
                    first_processed_ts = time.time()
                now = time.time()
                elapsed = max(1e-6, now - (first_processed_ts or now))
                inst_proc_fps = processed_frames_count / elapsed if elapsed > 0 else 0.0
                smoothed_proc_fps = inst_proc_fps if smoothed_proc_fps == 0.0 else (alpha * smoothed_proc_fps + (1 - alpha) * inst_proc_fps)
                inst_seen_fps = frames_seen / elapsed if elapsed > 0 else 0.0
                smoothed_seen_fps = inst_seen_fps if smoothed_seen_fps == 0.0 else (alpha * smoothed_seen_fps + (1 - alpha) * inst_seen_fps)
                # Throttle ETA reporting to ~3s
                if progress_callback is not None and total_frames and (now - last_report_time) >= 3.0:
                    try:
                        progress_callback(
                            min(0.999, frames_seen / max(1, total_frames)),
                            {
                                "frames_done": processed_frames_count,
                                "frames_total": max(1, total_frames),
                                "frames_seen": frames_seen,
                                "smoothed_proc_fps": smoothed_proc_fps,
                                "smoothed_seen_fps": smoothed_seen_fps,
                                "elapsed": elapsed,
                            },
                        )
                    except PoseExtractionCancelled:
                        raise
                    except Exception:
                        pass
                    last_report_time = now
            else:
                frame_data = {
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "box_conf": np.empty((0,), dtype=np.float32),
                    "keypoints": np.empty((0, 17, 2), dtype=np.float32),
                    "conf": np.empty((0, 17), dtype=np.float32),
                    "annotation_status": -1,
                }
            if not appended:
                pending.append(frame_data)

        _flush_batch()
        yield from pending
        pending.clear()

    def extract_pose_frames(
        self,
        video_path: str,
        confidence_threshold: float,
        start_time_seconds: int = 0,
        duration_seconds: int = 60,
        target_fps: int = 15,
        imgsz: Optional[int] = None,
        annotations_csv: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list:
        """In-memory pose extraction core: the full per-frame data list.

        Materializes :meth:`iter_pose_frames` (the streaming core). Kept for callers that
        want the whole list at once (file-writing wrapper + training scripts); the release
        path can consume the generator directly for bounded memory.
        """
        return list(self.iter_pose_frames(
            video_path=video_path,
            confidence_threshold=confidence_threshold,
            start_time_seconds=start_time_seconds,
            duration_seconds=duration_seconds,
            target_fps=target_fps,
            imgsz=imgsz,
            annotations_csv=annotations_csv,
            progress_callback=progress_callback,
        ))

    def extract_pose_data(
        self,
        video_path: str,
        confidence_threshold: float,
        start_time_seconds: int = 0,
        duration_seconds: int = 60,
        target_fps: int = 15,
        imgsz: Optional[int] = None,
        annotations_csv: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """File-writing wrapper around :meth:`extract_pose_frames`.

        Behaviour-preserving: computes the in-memory frame list via the core, then
        serializes it to the same NPZ path as before and returns that path. Kept for
        the training ``scripts/``; the release path calls the core directly.
        """
        all_frames_data = self.extract_pose_frames(
            video_path=video_path,
            confidence_threshold=confidence_threshold,
            start_time_seconds=start_time_seconds,
            duration_seconds=duration_seconds,
            target_fps=target_fps,
            imgsz=imgsz,
            annotations_csv=annotations_csv,
            progress_callback=progress_callback,
        )

        # output path
        if "yolov8" in self.model_path:
            model_size = self.model_path.split("yolov8")[1].split("-")[0]
        else:
            model_size = "s"
        subdir_name = f"yolo{model_size}_{confidence_threshold}conf_{target_fps}fps_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s"
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_posedata_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s_yolo{model_size}.npz"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_dir_path = os.path.join("pose_data", "raw", subdir_name)
            os.makedirs(output_dir_path, exist_ok=True)
            output_path = os.path.join(output_dir_path, output_filename)
        np.savez_compressed(output_path, frames=all_frames_data)
        return output_path
