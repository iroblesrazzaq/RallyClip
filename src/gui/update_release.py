"""GitHub Latest helpers for the in-app update button.

Frozen Mac builds download the arm64 DMG into ~/Downloads, verify SHA-256,
and open it. Source checkouts only need the release page URL. Never writes
into /Applications.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

GITHUB_REPO = "iroblesrazzaq/RallyClip"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases"
ASSET_DOWNLOAD_PREFIX = f"https://github.com/{GITHUB_REPO}/releases/download/"
DMG_NAME_RE = re.compile(r"^RallyClip-.+-macOS-arm64\.dmg$")
_HASH_CHUNK = 1024 * 1024
DOWNLOAD_TIMEOUT_SEC = 300
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def install_channel(*, frozen: Optional[bool] = None) -> str:
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    return "dmg" if frozen else "source"


def pick_macos_arm64_dmg_assets(assets: Any) -> tuple[Optional[str], Optional[str]]:
    """Return (dmg_url, sha256_url) for RallyClip-*-macOS-arm64.dmg.

    Skips artifact-rallyclip_* zips and non-arm64 disk images.
    """
    by_name: dict[str, str] = {}
    if not isinstance(assets, list):
        return None, None
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        url = str(asset.get("browser_download_url") or "")
        if not name or not url:
            continue
        if "artifact-rallyclip" in name.lower():
            continue
        by_name[name] = url
    dmg_name = next((name for name in by_name if DMG_NAME_RE.fullmatch(name)), None)
    if dmg_name is None:
        return None, None
    return by_name[dmg_name], by_name.get(f"{dmg_name}.sha256")


def parse_latest_release(payload: dict[str, Any]) -> dict[str, Any]:
    tag = str(payload.get("tag_name") or "").strip()
    version = tag[1:] if tag.startswith("v") else tag
    dmg_url, sha256_url = pick_macos_arm64_dmg_assets(payload.get("assets"))
    return {
        "latest_version": version or None,
        "latest_tag": tag or None,
        "release_url": payload.get("html_url") or GITHUB_RELEASES_URL,
        "release_name": payload.get("name") or tag or None,
        "dmg_url": dmg_url,
        "sha256_url": sha256_url,
    }


def is_allowed_asset_url(url: str) -> bool:
    return str(url).startswith(ASSET_DOWNLOAD_PREFIX)


def downloads_dir() -> Path:
    return (Path.home() / "Downloads").resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_dmg_sha256_sidecar(text: str, dmg_name: str) -> str:
    fallback: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rest = line.partition(" ")
        digest = digest.lower()
        if not _SHA256_RE.fullmatch(digest):
            continue
        name = Path(rest.strip().lstrip("*")).name if rest.strip() else ""
        if name == dmg_name:
            return digest
        fallback = digest
    if fallback:
        return fallback
    raise ValueError(f"checksum file did not contain a SHA-256 for {dmg_name}")


def open_downloaded_dmg(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True)
        return
    webbrowser.open(path.as_uri())


def _download_url(url: str, dest: Path, *, timeout: float = DOWNLOAD_TIMEOUT_SEC) -> None:
    if not is_allowed_asset_url(url):
        raise ValueError("refusing to download from an unexpected URL")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    request = Request(url, headers={"User-Agent": "RallyClip"})
    try:
        with urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def download_and_open_latest_dmg(latest: dict[str, Any]) -> dict[str, Any]:
    dmg_url = str(latest.get("dmg_url") or "")
    sha_url = str(latest.get("sha256_url") or "")
    if not dmg_url or not sha_url:
        raise ValueError("Latest GitHub release has no macOS arm64 DMG and checksum.")
    dmg_name = Path(urlparse(dmg_url).path).name
    if not DMG_NAME_RE.fullmatch(dmg_name):
        raise ValueError(f"unexpected DMG name: {dmg_name}")
    dest = downloads_dir() / dmg_name
    sha_path = dest.with_name(dest.name + ".sha256")
    _download_url(sha_url, sha_path)
    expected = parse_dmg_sha256_sidecar(sha_path.read_text(encoding="utf-8"), dmg_name)
    _download_url(dmg_url, dest)
    actual = sha256_file(dest)
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise ValueError("DMG checksum mismatch; download discarded.")
    open_downloaded_dmg(dest)
    logging.info("Opened downloaded update DMG %s", dest)
    return {
        "opened": True,
        "path": str(dest),
        "release_url": latest.get("release_url") or GITHUB_RELEASES_URL,
    }
