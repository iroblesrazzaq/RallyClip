from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import av
import h5py
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class YoloExtractConfig:
    model_path: str
    conf: float
    model_dir: Optional[str] = None
    device: Optional[str] = None
    batch_size: Optional[int] = None
    imgsz: int = 1920


class YoloHdf5Extractor:
    def __init__(self, cfg: YoloExtractConfig) -> None:
        self.cfg = cfg
        self.device = self._pick_device(cfg.device)
        self.batch_size = self._pick_batch_size(cfg.batch_size)
        self.model = self._load_model(cfg.model_path, cfg.model_dir)

    def extract(
        self,
        video_path: Path,
        output_path: Path,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        sampling_mode: str = "full_then_downsample",
        sample_fps: Optional[float] = None,
        overwrite: bool = False,
        resume: bool = True,
    ) -> Path:
        if sampling_mode not in {"full_then_downsample", "downsample_then_extract"}:
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
        if sampling_mode == "downsample_then_extract" and (sample_fps is None or sample_fps <= 0):
            raise ValueError("sample_fps must be > 0 when sampling_mode='downsample_then_extract'")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            with h5py.File(output_path, "r") as h5f:
                if h5f.attrs.get("complete", False):
                    logger.info("Skipping existing complete file: %s", output_path)
                    return output_path
            if not resume:
                logger.info("Skipping existing incomplete file (resume disabled): %s", output_path)
                return output_path

        end_time = None if duration is None else (start_time + duration)
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            fps_nominal = float(stream.average_rate) if stream.average_rate else 0.0

        last_frame_index = -1
        if output_path.exists() and resume:
            last_frame_index = self._read_last_frame_index(output_path)
            logger.info("Resuming from frame index %s for %s", last_frame_index, video_path)

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base
            total_frames = stream.frames or 0

            h5f = self._open_hdf5(
                output_path,
                video_path=video_path,
                width=width,
                height=height,
                fps_nominal=fps_nominal,
                start_time=start_time,
                duration=duration,
                sampling_mode=sampling_mode,
                sample_fps=sample_fps,
                overwrite=overwrite,
            )
            try:
                frame_iter = self._frame_iterator(container.decode(stream), time_base)
                iterator = tqdm(frame_iter, total=total_frames, desc="YOLO", unit="frame")
                batch_frames = []
                batch_meta = []
                sample_step = None if sample_fps is None else (1.0 / float(sample_fps))
                next_sample_ts = float(start_time)
                eps = 1e-9

                for frame_idx, frame_rgb, timestamp in iterator:
                    if timestamp < start_time:
                        continue
                    if end_time is not None and timestamp > end_time:
                        break
                    if frame_idx <= last_frame_index:
                        continue
                    if sampling_mode == "downsample_then_extract":
                        if sample_step is None:
                            continue
                        if timestamp + eps < next_sample_ts:
                            continue
                        while timestamp + eps >= next_sample_ts:
                            next_sample_ts += sample_step
                    batch_frames.append(frame_rgb)
                    batch_meta.append((frame_idx, timestamp))

                    if len(batch_frames) >= self.batch_size:
                        self._flush_batch(h5f, batch_frames, batch_meta)
                        batch_frames = []
                        batch_meta = []

                if batch_frames:
                    self._flush_batch(h5f, batch_frames, batch_meta)

                h5f.attrs["complete"] = True
                h5f.flush()
            finally:
                h5f.close()

        logger.info("Wrote YOLO HDF5: %s", output_path)
        return output_path

    def _flush_batch(self, h5f: h5py.File, batch_frames: list, batch_meta: list) -> None:
        results = self._predict(batch_frames)
        frames_group = h5f["frames"]
        det_group = h5f["detections"]

        for (frame_idx, timestamp), res in zip(batch_meta, results):
            boxes, box_conf, keypoints, kp_conf = self._extract_arrays(res)
            self._append_frame(
                frames_group,
                det_group,
                frame_idx=frame_idx,
                timestamp=timestamp,
                boxes=boxes,
                box_conf=box_conf,
                keypoints=keypoints,
                keypoint_conf=kp_conf,
            )

    def _predict(self, frames: list) -> list:
        try:
            return self.model.predict(
                source=frames,
                verbose=False,
                device=self.device,
                conf=self.cfg.conf,
                imgsz=self.cfg.imgsz,
                batch=self.batch_size,
            )
        except TypeError:
            return self.model.predict(
                source=frames,
                verbose=False,
                device=self.device,
                conf=self.cfg.conf,
                imgsz=self.cfg.imgsz,
            )

    @staticmethod
    def _extract_arrays(result) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if result is None or getattr(result, "boxes", None) is None:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 17, 2), dtype=np.float32),
                np.empty((0, 17), dtype=np.float32),
            )
        try:
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            box_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        except Exception:
            boxes = np.empty((0, 4), dtype=np.float32)
            box_conf = np.empty((0,), dtype=np.float32)
        try:
            keypoints = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
            kp_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)
        except Exception:
            keypoints = np.empty((0, 17, 2), dtype=np.float32)
            kp_conf = np.empty((0, 17), dtype=np.float32)
        return boxes, box_conf, keypoints, kp_conf

    @staticmethod
    def _append_frame(
        frames_group: h5py.Group,
        det_group: h5py.Group,
        frame_idx: int,
        timestamp: float,
        boxes: np.ndarray,
        box_conf: np.ndarray,
        keypoints: np.ndarray,
        keypoint_conf: np.ndarray,
    ) -> None:
        num_dets = int(boxes.shape[0])
        offsets = frames_group["frame_offsets"]
        det_count = int(offsets[-1])

        if num_dets > 0:
            _append_rows(det_group["boxes"], boxes)
            _append_rows(det_group["box_conf"], box_conf)
            _append_rows(det_group["keypoints"], keypoints)
            _append_rows(det_group["keypoint_conf"], keypoint_conf)
        offsets.resize((offsets.shape[0] + 1,))
        offsets[-1] = det_count + num_dets

        _append_rows(frames_group["frame_index"], np.asarray([frame_idx], dtype=np.int64))
        _append_rows(frames_group["timestamps"], np.asarray([timestamp], dtype=np.float64))

    @staticmethod
    def _frame_iterator(decoded: Iterable, time_base) -> Iterable[Tuple[int, np.ndarray, float]]:
        for idx, frame in enumerate(decoded):
            ts = frame.pts * time_base if frame.pts is not None else None
            if ts is None:
                continue
            yield idx, frame.to_ndarray(format="bgr24"), float(ts)

    def _open_hdf5(
        self,
        output_path: Path,
        video_path: Path,
        width: int,
        height: int,
        fps_nominal: float,
        start_time: float,
        duration: Optional[float],
        sampling_mode: str,
        sample_fps: Optional[float],
        overwrite: bool,
    ) -> h5py.File:
        mode = "w" if overwrite or not output_path.exists() else "a"
        h5f = h5py.File(output_path, mode)

        if "frames" in h5f and "detections" in h5f:
            self._validate_metadata(
                h5f,
                video_path,
                start_time,
                duration,
                self.cfg.model_path,
                self.cfg.conf,
                self.cfg.imgsz,
                sampling_mode,
                sample_fps,
            )
            return h5f

        h5f.attrs["video_path"] = str(video_path)
        h5f.attrs["yolo_model"] = self.cfg.model_path
        h5f.attrs["conf"] = float(self.cfg.conf)
        h5f.attrs["imgsz"] = int(self.cfg.imgsz)
        h5f.attrs["start_time"] = float(start_time)
        h5f.attrs["duration"] = -1.0 if duration is None else float(duration)
        h5f.attrs["sampling_mode"] = sampling_mode
        h5f.attrs["sample_fps"] = -1.0 if sample_fps is None else float(sample_fps)
        h5f.attrs["width"] = int(width)
        h5f.attrs["height"] = int(height)
        h5f.attrs["fps_nominal"] = float(fps_nominal)
        h5f.attrs["created_at"] = datetime.utcnow().isoformat() + "Z"
        h5f.attrs["complete"] = False

        frames = h5f.create_group("frames")
        detections = h5f.create_group("detections")

        frames.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
        frames.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype="f8", chunks=True)
        frames.create_dataset("frame_offsets", data=np.array([0], dtype=np.int64), maxshape=(None,), chunks=True)

        detections.create_dataset("boxes", shape=(0, 4), maxshape=(None, 4), dtype="f4", chunks=True, compression="gzip")
        detections.create_dataset("box_conf", shape=(0,), maxshape=(None,), dtype="f4", chunks=True, compression="gzip")
        detections.create_dataset(
            "keypoints",
            shape=(0, 17, 2),
            maxshape=(None, 17, 2),
            dtype="f4",
            chunks=True,
            compression="gzip",
        )
        detections.create_dataset(
            "keypoint_conf",
            shape=(0, 17),
            maxshape=(None, 17),
            dtype="f4",
            chunks=True,
            compression="gzip",
        )

        return h5f

    def _load_model(self, model_path: str, model_dir: Optional[str]) -> YOLO:
        yolo_arg = model_path
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            try:
                SETTINGS["weights_dir"] = os.path.abspath(model_dir)
            except Exception:
                pass
            candidate = Path(model_dir) / model_path
            if candidate.exists():
                yolo_arg = str(candidate)
        model = YOLO(yolo_arg)
        try:
            model.to(self.device)
        except Exception:
            pass
        return model

    @staticmethod
    def _pick_device(config_device: Optional[str]) -> str:
        if config_device:
            return config_device
        env_device = os.environ.get("POSE_DEVICE", "").strip().lower()
        if env_device in {"cpu", "cuda", "mps"}:
            return env_device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _pick_batch_size(config_batch: Optional[int]) -> int:
        env_bs = os.environ.get("POSE_BATCH_SIZE", "").strip()
        if env_bs.isdigit():
            return int(env_bs)
        if config_batch:
            return int(config_batch)
        return 8

    @staticmethod
    def _read_last_frame_index(output_path: Path) -> int:
        with h5py.File(output_path, "r") as h5f:
            if "frames" not in h5f:
                return -1
            frame_index = h5f["frames"]["frame_index"]
            if frame_index.shape[0] == 0:
                return -1
            return int(frame_index[-1])

    @staticmethod
    def _validate_metadata(
        h5f: h5py.File,
        video_path: Path,
        start_time: float,
        duration: Optional[float],
        model_path: str,
        conf: float,
        imgsz: int,
        sampling_mode: str = "full_rate",
        sample_fps: Optional[float] = None,
    ) -> None:
        expected = {
            "video_path": str(video_path),
            "start_time": float(start_time),
            "duration": -1.0 if duration is None else float(duration),
            "yolo_model": model_path,
            "conf": float(conf),
            "imgsz": int(imgsz),
            "sampling_mode": sampling_mode,
            "sample_fps": -1.0 if sample_fps is None else float(sample_fps),
        }
        for key, value in expected.items():
            if key not in h5f.attrs:
                continue
            if h5f.attrs[key] != value:
                raise ValueError(
                    f"Resume metadata mismatch for {key}: {h5f.attrs[key]} != {value}"
                )


def _append_rows(dataset: h5py.Dataset, data: np.ndarray) -> None:
    if data.size == 0:
        return
    new_size = dataset.shape[0] + data.shape[0]
    dataset.resize((new_size,) + dataset.shape[1:])
    dataset[-data.shape[0]:] = data
