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
                if features.shape[0] < seq_len:
                    logger.warning("Skipping %s (frames < seq_len)", video_name)
                    continue

                if self.cfg.split.strategy == "within_video":
                    ranges = temporal_split_indices(features.shape[0], self.cfg.split.val_ratio, self.cfg.split.test_ratio)
                    for split_name, (start, end) in ranges.items():
                        seqs, labels = _make_sequences(features[start:end], targets[start:end], seq_len, overlap)
                        if seqs:
                            all_splits[split_name].append((seqs, labels, video_name))
                    manifest["videos"][video_name] = {"total_frames": int(features.shape[0])}
                else:
                    split_bucket = "train"
                    if video_name in video_split.val:
                        split_bucket = "val"
                    if video_name in video_split.test:
                        split_bucket = "test"
                    seqs, labels = _make_sequences(features, targets, seq_len, overlap)
                    if seqs:
                        all_splits[split_bucket].append((seqs, labels, video_name))
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
            features_arr, targets_arr, video_ids = _merge_sequences(datasets)
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
            manifest["splits"][split_name] = list({v for _, _, v in datasets})

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


def _make_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int, overlap: int):
    sequences = []
    labels = []
    step = max(1, seq_len - overlap)
    for start in range(0, features.shape[0] - seq_len + 1, step):
        end = start + seq_len
        sequences.append(features[start:end])
        labels.append(targets[start:end])
    return sequences, labels


def _concat_features(split_data):
    all_features = []
    for seqs, _, _ in split_data:
        all_features.append(np.asarray(seqs))
    if not all_features:
        return np.array([])
    return np.concatenate(all_features, axis=0)


def _merge_sequences(split_data):
    if not split_data:
        return np.array([]), np.array([]), np.array([])

    all_features = []
    all_targets = []
    all_video_ids = []
    video_map = {}
    for seqs, labels, video in split_data:
        if video not in video_map:
            video_map[video] = len(video_map)
        vid = video_map[video]
        for seq, label in zip(seqs, labels):
            all_features.append(seq)
            all_targets.append(label)
            all_video_ids.append(vid)
    return (
        np.asarray(all_features, dtype=np.float32),
        np.asarray(all_targets, dtype=np.int8),
        np.asarray(all_video_ids, dtype=np.int32),
    )
