import os
import subprocess
import time
from typing import Callable, Optional

import numpy as np
import av
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import torch
from tqdm import tqdm
import logging
import os


class PoseExtractionCancelled(Exception):
    """Raised when upstream caller requests pose extraction cancellation."""


class PoseExtractor:
    """
    YOLOv8-pose based extractor that converts input video frames into pose arrays
    and saves compressed npz artifacts. Designed for reuse by CLI and GUI.
    """

    def __init__(self, model_dir: Optional[str] = None, model_path: str = "yolov8s-pose.pt") -> None:
        # Set early so downstream checks can use it
        self.model_path = model_path
        self.model_dir = model_dir

        profile = os.environ.get("PIPELINE_PROFILE", "").strip().lower()
        if not profile:
            try:
                git_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True)
                branch_name = git_branch.stdout.strip()
                profile = "mvp" if branch_name == "mvp" else "main"
            except Exception:
                profile = "main"

        env_device = os.environ.get("POSE_DEVICE", "").strip().lower()
        valid_devices = {"cpu", "cuda", "mps"}
        device_from_env = env_device in valid_devices
        if device_from_env:
            self.device = env_device
        else:
            if profile == "mvp":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"

        # Work around known MPS pose performance bug unless user explicitly requests it.
        if (not device_from_env) and self.device == "mps" and "pose" in self.model_path.lower():
            logging.warning("POSE: defaulting to cpu due to known MPS pose performance issues; set POSE_DEVICE=mps to force MPS.")
            self.device = "cpu"

        env_bs = os.environ.get("POSE_BATCH_SIZE", "").strip()
        if env_bs.isdigit():
            self.batch_size = int(env_bs)
        else:
            if profile == "mvp":
                self.batch_size = 1 if self.device == "cpu" else 16
            else:
                if self.device == "mps":
                    self.batch_size = 2
                elif self.device == "cpu":
                    self.batch_size = 1
                elif self.device == "cuda":
                    self.batch_size = 8
                else:
                    self.batch_size = 1

        # Prefer local file if model_dir provided and file exists; otherwise let
        # Ultralytics handle download from model name (e.g., "yolov8s-pose.pt").
        yolo_arg = self.model_path
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            # Direct ultralytics downloads into the provided models directory
            try:
                SETTINGS["weights_dir"] = os.path.abspath(model_dir)
            except Exception:
                pass
            candidate = os.path.join(model_dir, self.model_path)
            if os.path.exists(candidate):
                yolo_arg = candidate
        self.model = YOLO(yolo_arg)
        try:
            self.model.to(self.device)
        except Exception:
            pass

    def frame_iterator_pyav(self, video_path: str):
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                time_base = stream.time_base
                for frame in container.decode(stream):
                    ts = float(frame.pts * time_base) if frame.pts is not None else None
                    if ts is None:
                        continue
                    yield frame.to_ndarray(format="bgr24"), ts
        except Exception as e:
            logging.error("[PyAV Error] %s", e)
            return

    def extract_pose_data(
        self,
        video_path: str,
        confidence_threshold: float,
        start_time_seconds: int = 0,
        duration_seconds: int = 60,
        target_fps: int = 15,
        annotations_csv: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        import pandas as pd

        annotations = None
        if annotations_csv and annotations_csv != "None" and os.path.exists(annotations_csv):
            try:
                annotations = pd.read_csv(annotations_csv)
            except Exception as e:
                print(f"Warning: could not read annotations {annotations_csv}: {e}")

        try:
            with av.open(video_path) as container:
                total_frames = container.streams.video[0].frames
        except Exception:
            total_frames = 0

        all_frames_data = []
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
            starts = annotations["start_time"].to_numpy(dtype=float)
            ends = annotations["end_time"].to_numpy(dtype=float)
            annotation_index = 0
            num_annotations = starts.size
        else:
            starts = ends = None
            annotation_index = 0
            num_annotations = 0

        batch_frames = []
        batch_indices = []

        def _flush_batch():
            nonlocal batch_frames, batch_indices
            if not batch_frames:
                return
            try:
                results = self.model.predict(
                    source=batch_frames,
                    verbose=False,
                    device=self.device,
                    conf=confidence_threshold,
                    imgsz=1920,
                    batch=self.batch_size,
                )
            except TypeError:
                results = self.model.predict(
                    source=batch_frames,
                    verbose=False,
                    device=self.device,
                    conf=confidence_threshold,
                    imgsz=1920,
                )
            for i, res in enumerate(results):
                idx = batch_indices[i]
                frame_data = {}
                if res is not None and getattr(res, "boxes", None) is not None:
                    try:
                        frame_data["boxes"] = res.boxes.xyxy.detach().cpu().numpy()
                    except Exception:
                        frame_data["boxes"] = np.array([])
                    try:
                        frame_data["keypoints"] = res.keypoints.xy.detach().cpu().numpy()
                        frame_data["conf"] = res.keypoints.conf.detach().cpu().numpy()
                    except Exception:
                        frame_data["keypoints"] = np.array([])
                        frame_data["conf"] = np.array([])
                else:
                    frame_data["boxes"] = np.array([])
                    frame_data["keypoints"] = np.array([])
                    frame_data["conf"] = np.array([])
                frame_data["annotation_status"] = all_frames_data[idx].get("annotation_status", 0)
                all_frames_data[idx] = frame_data
            batch_frames = []
            batch_indices = []

        gen = self.frame_iterator_pyav(video_path)
        show_tqdm = os.environ.get("RALLYVISION_NO_TQDM", "").strip().lower() not in {"1", "true", "yes"}
        iterator = tqdm(gen, total=total_frames, desc="Processing frames", unit="frame") if show_tqdm else gen
        # simple progress proxy if tqdm not desired in GUI
        for frame, current_timestamp in iterator:
            frames_seen += 1
            appended = False
            annotation_status_current = -100
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
                all_frames_data.append({
                    "boxes": np.array([]),
                    "keypoints": np.array([]),
                    "conf": np.array([]),
                    "annotation_status": annotation_status_current,
                })
                batch_frames.append(frame)
                batch_indices.append(len(all_frames_data) - 1)
                appended = True
                if len(batch_frames) >= self.batch_size:
                    _flush_batch()
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
                frame_data = {"boxes": np.array([]), "keypoints": np.array([]), "conf": np.array([]), "annotation_status": -100}
            if not appended:
                all_frames_data.append(frame_data)

        _flush_batch()

        # output path
        if "yolov8" in self.model_path:
            model_size = self.model_path.split("yolov8")[1].split("-")[0]
        else:
            model_size = "s"
        subdir_name = f"yolo{model_size}_{confidence_threshold}conf_{target_fps}fps_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s"
        output_dir = os.path.join("pose_data", "raw", subdir_name)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_posedata_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s_yolo{model_size}.npz"
        output_path = os.path.join(output_dir, output_filename)
        np.savez_compressed(output_path, frames=all_frames_data)
        return output_path
