"""Pack, verify, and fetch the shipped RallyClip ONNX artifact.

Git tracks manifests and SHA256SUMS. Weight files live on GitHub Release
``artifact-rallyclip_v0.5.0``. From-source checkouts unpack into
``DEFAULT_ARTIFACT_DIR``. Frozen apps embed that folder at build time and
must not call this at launch.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional

from runtime.defaults import DEFAULT_ARTIFACT_DIR

GITHUB_REPO = "iroblesrazzaq/RallyClip"
ARTIFACT_TAG = "artifact-rallyclip_v0.5.0"
ARTIFACT_ZIP_NAME = "rallyclip_v0.5.0.zip"
REQUIRED_FILES = (
    "model.onnx",
    "scaler.json",
    "manifest.json",
    "yolov8n-pose-960-dynamic.onnx",
    "yolov8n-pose-544x960-static.onnx",
)
SHA256SUMS_NAME = "SHA256SUMS"
_HASH_CHUNK = 1024 * 1024


class ArtifactError(RuntimeError):
    """Checksum or download failure; callers should fail closed."""


def artifact_dir(repo_root: Path) -> Path:
    return (repo_root / DEFAULT_ARTIFACT_DIR).resolve()


def default_download_url() -> str:
    return (
        f"https://github.com/{GITHUB_REPO}/releases/download/"
        f"{ARTIFACT_TAG}/{ARTIFACT_ZIP_NAME}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(dest_dir: Path, names: Iterable[str] = REQUIRED_FILES) -> Path:
    lines = []
    for name in names:
        path = dest_dir / name
        if not path.is_file():
            raise ArtifactError(f"cannot checksum missing file: {path}")
        lines.append(f"{sha256_file(path)}  {name}\n")
    sums_path = dest_dir / SHA256SUMS_NAME
    sums_path.write_text("".join(lines), encoding="utf-8")
    return sums_path


def parse_sha256sums(text: str) -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        expected[name] = digest.lower()
    return expected


def load_sha256sums(dest_dir: Path) -> dict[str, str]:
    sums_path = dest_dir / SHA256SUMS_NAME
    if not sums_path.is_file():
        raise ArtifactError(f"missing {SHA256SUMS_NAME} under {dest_dir}")
    return parse_sha256sums(sums_path.read_text(encoding="utf-8"))


def missing_required_files(dest_dir: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (dest_dir / name).is_file()]


def verify_artifact_dir(dest_dir: Path, expected: Optional[dict[str, str]] = None) -> None:
    missing = missing_required_files(dest_dir)
    if missing:
        raise ArtifactError(f"missing {missing} under {dest_dir}")
    checksums = expected if expected is not None else load_sha256sums(dest_dir)
    for name in REQUIRED_FILES:
        digest = checksums.get(name)
        if digest is None:
            raise ArtifactError(f"{SHA256SUMS_NAME} has no entry for {name}")
        actual = sha256_file(dest_dir / name)
        if actual != digest:
            raise ArtifactError(f"hash mismatch for {name}: expected {digest}, got {actual}")


def artifact_complete(dest_dir: Path) -> bool:
    try:
        verify_artifact_dir(dest_dir)
    except ArtifactError:
        return False
    return True


def pack_artifact_dir(src_dir: Path, zip_path: Path) -> Path:
    missing = missing_required_files(src_dir)
    if missing:
        raise ArtifactError(f"cannot pack; missing {missing} under {src_dir}")
    write_sha256sums(src_dir)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    names = [*REQUIRED_FILES, SHA256SUMS_NAME]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(src_dir / name, arcname=name)
    (zip_path.parent / f"{zip_path.name}.sha256").write_text(
        f"{sha256_file(zip_path)}  {zip_path.name}\n",
        encoding="utf-8",
    )
    return zip_path


def pack_default_artifact(repo_root: Path, out_dir: Path) -> Path:
    zip_path = out_dir / ARTIFACT_ZIP_NAME
    return pack_artifact_dir(artifact_dir(repo_root), zip_path)


def unpack_artifact_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            if name.endswith("/") or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ArtifactError(f"refusing unsafe zip member {name!r}")
        zf.extractall(dest_dir)


def _download(url: str, dest: Path) -> None:
    headers = {"User-Agent": "RallyClip-artifact-fetch"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            dest.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ArtifactError(f"failed to download {url}: {exc}") from exc


def fetch_artifact(
    dest_dir: Path,
    *,
    url: Optional[str] = None,
    zip_path: Optional[Path] = None,
) -> Path:
    """Ensure ``dest_dir`` has the shipped weights. No-op when hashes already match."""
    dest_dir = dest_dir.resolve()
    if artifact_complete(dest_dir):
        return dest_dir

    expected: Optional[dict[str, str]] = None
    sums_path = dest_dir / SHA256SUMS_NAME
    if sums_path.is_file():
        expected = parse_sha256sums(sums_path.read_text(encoding="utf-8"))

    if zip_path is None:
        download_url = url or os.environ.get("RALLYCLIP_ARTIFACT_URL") or default_download_url()
        with tempfile.TemporaryDirectory(prefix="rallyclip-artifact-") as tmp:
            tmp_zip = Path(tmp) / ARTIFACT_ZIP_NAME
            _download(download_url, tmp_zip)
            unpack_artifact_zip(tmp_zip, dest_dir)
    else:
        unpack_artifact_zip(zip_path, dest_dir)

    verify_artifact_dir(dest_dir, expected=expected)
    return dest_dir


def fetch_default_artifact(repo_root: Path) -> Path:
    dest = artifact_dir(repo_root)
    committed_sums = dest / SHA256SUMS_NAME
    expected = None
    if committed_sums.is_file():
        expected = parse_sha256sums(committed_sums.read_text(encoding="utf-8"))
    if missing_required_files(dest) == [] and expected is not None:
        verify_artifact_dir(dest, expected=expected)
        return dest
    if artifact_complete(dest):
        return dest
    zip_override = os.environ.get("RALLYCLIP_ARTIFACT_ZIP")
    zip_path = Path(zip_override).expanduser().resolve() if zip_override else None
    fetch_artifact(dest, zip_path=zip_path)
    if expected is not None:
        verify_artifact_dir(dest, expected=expected)
    return dest


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pack or fetch the RallyClip ONNX artifact.")
    sub = parser.add_subparsers(dest="command", required=True)
    pack_p = sub.add_parser("pack", help="Zip DEFAULT_ARTIFACT_DIR (weights must already exist).")
    pack_p.add_argument("--repo-root", default=".")
    pack_p.add_argument("out_dir", nargs="?", default="dist")
    fetch_p = sub.add_parser("fetch", help="Download the release zip if local weights are incomplete.")
    fetch_p.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            zip_path = pack_default_artifact(Path(args.repo_root).resolve(), Path(args.out_dir).resolve())
            print(zip_path)
            return 0
        if args.command == "fetch":
            dest = fetch_default_artifact(Path(args.repo_root).resolve())
            print(dest)
            return 0
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
