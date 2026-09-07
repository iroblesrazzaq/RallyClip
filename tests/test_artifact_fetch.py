from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from runtime.artifact import (
    ARTIFACT_ZIP_NAME,
    REQUIRED_FILES,
    SHA256SUMS_NAME,
    ArtifactError,
    artifact_complete,
    fetch_artifact,
    pack_artifact_dir,
    sha256_file,
    unpack_artifact_zip,
    verify_artifact_dir,
    write_sha256sums,
)


def _seed_artifact(dest: Path, payload: bytes = b"onnx-bytes") -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (dest / name).write_bytes(payload if name.endswith(".onnx") else b'{"ok": true}')
    write_sha256sums(dest)


def test_pack_and_unpack_roundtrip_verifies_checksums(tmp_path: Path):
    src = tmp_path / "src"
    _seed_artifact(src, b"weight-v1")
    zip_path = tmp_path / ARTIFACT_ZIP_NAME
    pack_artifact_dir(src, zip_path)
    assert zip_path.is_file()
    assert (tmp_path / f"{ARTIFACT_ZIP_NAME}.sha256").is_file()
    sidecar = (tmp_path / f"{ARTIFACT_ZIP_NAME}.sha256").read_text(encoding="utf-8")
    assert sha256_file(zip_path) in sidecar

    dest = tmp_path / "dest"
    unpack_artifact_zip(zip_path, dest)
    verify_artifact_dir(dest)
    assert artifact_complete(dest)


def test_verify_fails_closed_on_hash_mismatch(tmp_path: Path):
    dest = tmp_path / "art"
    _seed_artifact(dest, b"original")
    (dest / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        verify_artifact_dir(dest)
    assert artifact_complete(dest) is False


def test_fetch_is_noop_when_checksums_already_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = tmp_path / "art"
    _seed_artifact(dest, b"local")

    def boom(*_args, **_kwargs):
        raise AssertionError("must not download when artifact is complete")

    monkeypatch.setattr("runtime.artifact._download", boom)
    assert fetch_artifact(dest) == dest.resolve()


def test_fetch_from_local_zip_when_incomplete(tmp_path: Path):
    src = tmp_path / "src"
    _seed_artifact(src, b"from-zip")
    zip_path = tmp_path / "payload.zip"
    pack_artifact_dir(src, zip_path)

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / SHA256SUMS_NAME).write_text((src / SHA256SUMS_NAME).read_text(encoding="utf-8"), encoding="utf-8")
    fetched = fetch_artifact(dest, zip_path=zip_path)
    assert fetched == dest.resolve()
    assert (dest / "model.onnx").read_bytes() == b"from-zip"
    verify_artifact_dir(dest)


def test_fetch_refuses_unsafe_zip_members(tmp_path: Path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.onnx", b"nope")
    with pytest.raises(ArtifactError, match="unsafe"):
        unpack_artifact_zip(zip_path, tmp_path / "out")
