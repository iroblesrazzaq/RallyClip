from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from training.dataset.splits import SplitConfig, split_videos, temporal_split_indices

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    seq_len_seconds: float
    overlap_seconds: float
    target_fps: float
    split: SplitConfig


class DatasetBuilder:
    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg

    def build(self, feature_root: Path, output_dir: Path, video_list: List[str], feature_set: str) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        seq_len = max(1, int(round(self.cfg.seq_len_seconds * self.cfg.target_fps)))
        overlap = max(0, int(round(self.cfg.overlap_seconds * self.cfg.target_fps)))
        split_strategy = self.cfg.split.strategy
        held_out_video = self._validate_loso_setup(video_list)

        video_split = split_videos(video_list, self.cfg.split)
        all_splits = {"train": [], "val": [], "test": []}
        manifest: Dict[str, Dict] = {"splits": {}, "videos": {}, "feature_set": feature_set}

        for split_name in ("train", "val", "test"):
            manifest["splits"][split_name] = []

        for video_name in video_list:
            feature_path = feature_root / f"{Path(video_name).stem}__features__{feature_set}.h5"
            if not feature_path.exists():
                logger.warning("Missing features for %s", video_name)
                continue

            with h5py.File(feature_path, "r") as h5f:
                features = h5f["features"][:]
                targets = h5f["targets"][:]
                frame_index = h5f["frame_index"][:] if "frame_index" in h5f else np.arange(features.shape[0], dtype=np.int64)
                timestamps = (
                    h5f["timestamps"][:]
                    if "timestamps" in h5f
                    else (np.arange(features.shape[0], dtype=np.float64) / max(1e-6, self.cfg.target_fps))
                )
                if features.shape[0] < seq_len:
                    logger.warning("Skipping %s (frames < seq_len)", video_name)
                    continue

                if split_strategy == "within_video":
                    ranges = temporal_split_indices(features.shape[0], self.cfg.split.val_ratio, self.cfg.split.test_ratio)
                    for split_name, (start, end) in ranges.items():
                        seqs, labels, seq_frame_idx, seq_times = _make_sequences(
                            features[start:end],
                            targets[start:end],
                            frame_index[start:end],
                            timestamps[start:end],
                            seq_len,
                            overlap,
                        )
                        if seqs:
                            all_splits[split_name].append((seqs, labels, seq_frame_idx, seq_times, video_name))
                    manifest["videos"][video_name] = {"total_frames": int(features.shape[0])}
                elif split_strategy == "loso_temporal_val":
                    if video_name == held_out_video:
                        seqs, labels, seq_frame_idx, seq_times = _make_sequences(
                            features,
                            targets,
                            frame_index,
                            timestamps,
                            seq_len,
                            overlap,
                        )
                        if seqs:
                            all_splits["test"].append((seqs, labels, seq_frame_idx, seq_times, video_name))
                        manifest["videos"][video_name] = {
                            "total_frames": int(features.shape[0]),
                            "split": "test",
                            "held_out": True,
                        }
                    else:
                        ranges = temporal_split_indices(features.shape[0], self.cfg.split.val_ratio, 0.0)
                        for split_name in ("train", "val"):
                            start, end = ranges[split_name]
                            seqs, labels, seq_frame_idx, seq_times = _make_sequences(
                                features[start:end],
                                targets[start:end],
                                frame_index[start:end],
                                timestamps[start:end],
                                seq_len,
                                overlap,
                            )
                            if seqs:
                                all_splits[split_name].append((seqs, labels, seq_frame_idx, seq_times, video_name))
                        manifest["videos"][video_name] = {
                            "total_frames": int(features.shape[0]),
                            "split": "train_val_temporal",
                            "train_end_frame_index": int(ranges["train"][1]),
                            "val_start_frame_index": int(ranges["val"][0]),
                        }
                else:
                    split_bucket = "train"
                    if video_name in video_split.val:
                        split_bucket = "val"
                    if video_name in video_split.test:
                        split_bucket = "test"
                    seqs, labels, seq_frame_idx, seq_times = _make_sequences(
                        features, targets, frame_index, timestamps, seq_len, overlap
                    )
                    if seqs:
                        all_splits[split_bucket].append((seqs, labels, seq_frame_idx, seq_times, video_name))
                    manifest["videos"][video_name] = {"total_frames": int(features.shape[0]), "split": split_bucket}

        scaler = StandardScaler()
        train_features = _concat_features(all_splits["train"])
        if train_features.size == 0:
            logger.warning("No training data available")
            return None

        scaler.fit(train_features.reshape(-1, train_features.shape[-1]))
        scaler_path = output_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)

        for split_name, datasets in all_splits.items():
            features_arr, targets_arr, video_ids, seq_frame_idx, seq_times, video_names = _merge_sequences(datasets)
            if features_arr.size == 0:
                logger.warning("No data for split %s", split_name)
                continue
            flat = features_arr.reshape(-1, features_arr.shape[-1])
            scaled = scaler.transform(flat).reshape(features_arr.shape)

            out_path = output_dir / f"{split_name}.h5"
            with h5py.File(out_path, "w") as out:
                out.create_dataset("features", data=scaled, compression="gzip")
                out.create_dataset("targets", data=targets_arr)
                out.create_dataset("sequence_video_index", data=video_ids, dtype="i4")
                out.create_dataset("sequence_frame_index", data=seq_frame_idx, dtype="i8")
                out.create_dataset("sequence_timestamps", data=seq_times, dtype="f8")
                out.create_dataset(
                    "video_index_to_name",
                    data=np.asarray(video_names, dtype=h5py.string_dtype(encoding="utf-8")),
                )
            manifest["splits"][split_name] = sorted({v for _, _, _, _, v in datasets})

        manifest_path = output_dir / "dataset_manifest.json"
        manifest["config"] = {
            "seq_len_seconds": self.cfg.seq_len_seconds,
            "overlap_seconds": self.cfg.overlap_seconds,
            "target_fps": self.cfg.target_fps,
            "split": self.cfg.split.__dict__,
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        logger.info("Dataset built at %s", output_dir)
        return output_dir

    def _validate_loso_setup(self, video_list: List[str]) -> Optional[str]:
        if self.cfg.split.strategy != "loso_temporal_val":
            return None

        if len(self.cfg.split.test_videos) != 1:
            raise ValueError("loso_temporal_val requires exactly one entry in dataset.split.test_videos")
        held_out = self.cfg.split.test_videos[0]
        if held_out not in video_list:
            raise ValueError(f"Held-out test video not found in selected videos: {held_out}")
        if not (0.0 < float(self.cfg.split.val_ratio) < 1.0):
            raise ValueError("dataset.split.val_ratio must be in (0, 1) for loso_temporal_val")
        return held_out


def _make_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    frame_index: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int,
    overlap: int,
):
    sequences = []
    labels = []
    seq_frame_idx = []
    seq_times = []
    step = max(1, seq_len - overlap)
    for start in range(0, features.shape[0] - seq_len + 1, step):
        end = start + seq_len
        sequences.append(features[start:end])
        labels.append(targets[start:end])
        seq_frame_idx.append(frame_index[start:end])
        seq_times.append(timestamps[start:end])
    return sequences, labels, seq_frame_idx, seq_times


def _concat_features(split_data):
    all_features = []
    for seqs, _, _, _, _ in split_data:
        all_features.append(np.asarray(seqs))
    if not all_features:
        return np.array([])
    return np.concatenate(all_features, axis=0)


def _merge_sequences(split_data):
    if not split_data:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []

    all_features = []
    all_targets = []
    all_video_ids = []
    all_frame_idx = []
    all_times = []
    video_map = {}
    for seqs, labels, seq_frame_idx, seq_times, video in split_data:
        if video not in video_map:
            video_map[video] = len(video_map)
        vid = video_map[video]
        for seq, label, frame_idx, times in zip(seqs, labels, seq_frame_idx, seq_times):
            all_features.append(seq)
            all_targets.append(label)
            all_video_ids.append(vid)
            all_frame_idx.append(frame_idx)
            all_times.append(times)
    video_names = [name for name, _ in sorted(video_map.items(), key=lambda item: item[1])]
    return (
        np.asarray(all_features, dtype=np.float32),
        np.asarray(all_targets, dtype=np.int8),
        np.asarray(all_video_ids, dtype=np.int32),
        np.asarray(all_frame_idx, dtype=np.int64),
        np.asarray(all_times, dtype=np.float64),
        video_names,
    )
