"""Cache fingerprint helpers for preprocessed / feature HDF5 files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_dict(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_preprocess_fingerprint(
    *,
    annotations_path: Path,
    raw_h5_path: Path,
    target_fps: float,
    court_model_path: str,
    court_target_time: int,
) -> Dict[str, Any]:
    return {
        "kind": "preprocess_v1",
        "annotations_sha256": file_sha256(annotations_path),
        "raw_h5_sha256": file_sha256(raw_h5_path),
        "target_fps": float(target_fps),
        "court_model_path": str(court_model_path),
        "court_target_time": int(court_target_time),
    }


def build_features_fingerprint(
    *,
    preproc_h5_path: Path,
    feature_set: str,
    target_fps: float,
) -> Dict[str, Any]:
    return {
        "kind": "features_v1",
        "preproc_sha256": file_sha256(preproc_h5_path),
        "feature_set": str(feature_set),
        "target_fps": float(target_fps),
    }


def read_h5_fingerprint(path: Path) -> Optional[str]:
    try:
        import h5py

        with h5py.File(path, "r") as h5f:
            value = h5f.attrs.get("fingerprint")
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
    except Exception:
        return None


def tmp_path_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{uuid4().hex}")
