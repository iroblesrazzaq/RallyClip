"""Manifest-defined pose model backends.

A pose backend is a directory bundling ONNX model(s) with a `manifest.json`
describing the full runtime contract (input/letterbox, head family, keypoint
schema, sha256). Training extraction and the desktop app both resolve models
through manifests so a dataset's identity is pinned to an exact artifact:
the cache tag is `<name>@<sha8>`, never a loose filename.

Execution provider is provenance, not identity: the tag is always derived from
the primary (dynamic) model's sha, regardless of which provider/sibling runs.
The CoreML path uses the static-shape sibling (dynamic axes block the ANE) and
is only offered for head families whose decode is EP-robust (v8 raw head; the
YOLO26 e2e head mis-selects candidates under fp16 — measured 30-180px errors).

Torch-free: numpy + onnxruntime via yolo_onnx_runner.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from extraction.yolo_onnx_runner import YOLO

SUPPORTED_HEAD_FAMILIES = {"v8-raw-head", "yolo26-e2e"}
COREML_SAFE_HEAD_FAMILIES = {"v8-raw-head"}


@dataclass(frozen=True)
class PoseBackendMeta:
    name: str
    head_family: str
    imgsz: int
    model_path: Path
    model_sha256: str
    contract_version: int
    static_model_path: Optional[Path] = None
    static_model_sha256: Optional[str] = None

    @property
    def tag(self) -> str:
        """Cache/dataset identity tag: name@sha8 (always the primary model)."""
        return f"{self.name}@{self.model_sha256[:8]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_model_path(manifest_dir: Path, filename: str, expected_sha: str) -> Path:
    path = manifest_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Pose model missing: {path}")
    actual = _sha256(path)
    if actual != expected_sha:
        raise ValueError(
            f"Pose model sha mismatch for {path}: manifest says "
            f"{expected_sha[:12]}…, file is {actual[:12]}…"
        )
    return path


def load_manifest(manifest_path: Path) -> PoseBackendMeta:
    """Parse and validate a pose manifest; verifies model file sha256s."""
    manifest_path = Path(manifest_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    head_family = manifest["head_family"]
    if head_family not in SUPPORTED_HEAD_FAMILIES:
        raise ValueError(
            f"Unsupported head_family {head_family!r} in {manifest_path}; "
            f"expected one of {sorted(SUPPORTED_HEAD_FAMILIES)}"
        )
    model_path = _verified_model_path(
        manifest_path.parent, manifest["model_file"], manifest["model_sha256"]
    )
    static_model_path = None
    static_sha = manifest.get("static_model_sha256")
    if manifest.get("static_model_file"):
        static_model_path = _verified_model_path(
            manifest_path.parent, manifest["static_model_file"], static_sha
        )
    return PoseBackendMeta(
        name=manifest["name"],
        head_family=head_family,
        imgsz=int(manifest["input"]["imgsz"]),
        model_path=model_path,
        model_sha256=manifest["model_sha256"],
        contract_version=int(manifest["contract_version"]),
        static_model_path=static_model_path,
        static_model_sha256=static_sha,
    )


def load_pose_backend(
    manifest_path: Path, *, provider: str = "cpu"
) -> tuple[YOLO, PoseBackendMeta]:
    """Load an ONNX pose backend from a manifest.

    provider: "cpu" runs the primary (dynamic) model on the CPU EP; "coreml"
    runs the static-shape sibling on CoreMLExecutionProvider (CPU fallback for
    unsupported nodes). CoreML is refused for head families where fp16 breaks
    decode fidelity.
    """
    meta = load_manifest(manifest_path)
    if provider == "cpu":
        return YOLO(str(meta.model_path)), meta
    if provider == "coreml":
        if meta.head_family not in COREML_SAFE_HEAD_FAMILIES:
            raise ValueError(
                f"CoreML provider refused for head_family {meta.head_family!r}: "
                "fp16 candidate selection is not numerically faithful. Use cpu."
            )
        if meta.static_model_path is None:
            raise ValueError(
                f"Manifest {meta.name} has no static_model_file; CoreML needs "
                "a static-shape sibling (dynamic axes block the ANE)."
            )
        model = YOLO(
            str(meta.static_model_path),
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        return model, meta
    raise ValueError(f"Unknown pose provider {provider!r}: expected cpu or coreml")
