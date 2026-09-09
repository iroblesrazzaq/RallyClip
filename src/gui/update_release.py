"""GitHub Latest helpers for the in-app update button.

Frozen Mac builds download the arm64 DMG into ~/Downloads, verify SHA-256,
and open it. Source checkouts only need the release page URL. Never writes
into /Applications.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import sys
import threading
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

GITHUB_REPO = "iroblesrazzaq/RallyClip"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases"
ASSET_DOWNLOAD_PREFIX = f"https://github.com/{GITHUB_REPO}/releases/download/"
DMG_NAME_RE = re.compile(r"^RallyClip-.+-macOS-arm64\.dmg$")
APP_TAG_RE = re.compile(r"^v\d")
_HASH_CHUNK = 1024 * 1024
DOWNLOAD_TIMEOUT_SEC = 300
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOWNLOAD_GUARD = threading.Lock()
_DOWNLOAD_GENERATION = 0
_DOWNLOAD_CANCEL = threading.Event()


class UpdateDownloadCancelled(Exception):
    """Raised when the user cancels an in-flight DMG download."""


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


def is_published_app_release(payload: Any) -> bool:
    """True for a published app `v*` release, not an inference-artifact tag."""
    if not isinstance(payload, dict):
        return False
    if payload.get("draft") or payload.get("prerelease"):
        return False
    tag = str(payload.get("tag_name") or "").strip()
    lowered = tag.lower()
    if lowered.startswith("artifact-") or "artifact-rallyclip" in lowered:
        return False
    return APP_TAG_RE.match(tag) is not None


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


def select_latest_app_release(releases: Any) -> Optional[dict[str, Any]]:
    """Newest published `v*` app release from a GitHub `/releases` list.

    GitHub `/releases/latest` is whichever non-draft, non-prerelease was
    published last, including `artifact-rallyclip_*` model zips.
    """
    if not isinstance(releases, list):
        return None
    candidates = [item for item in releases if is_published_app_release(item)]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: str(item.get("published_at") or item.get("created_at") or ""),
        reverse=True,
    )
    return parse_latest_release(candidates[0])


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
    raise ValueError(f"checksum file did not contain a SHA-256 for {dmg_name}")


def request_update_cancel() -> None:
    _DOWNLOAD_CANCEL.set()


def begin_update_download() -> tuple[int, threading.Event]:
    """Invalidate any in-flight download and return a fresh cancel event."""
    global _DOWNLOAD_GENERATION, _DOWNLOAD_CANCEL
    with _DOWNLOAD_GUARD:
        _DOWNLOAD_CANCEL.set()
        _DOWNLOAD_GENERATION += 1
        _DOWNLOAD_CANCEL = threading.Event()
        return _DOWNLOAD_GENERATION, _DOWNLOAD_CANCEL


def _raise_if_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise UpdateDownloadCancelled("Update download cancelled.")


def open_downloaded_dmg(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True)
        return
    webbrowser.open(path.as_uri())


def _download_url(
    url: str,
    dest: Path,
    *,
    timeout: float = DOWNLOAD_TIMEOUT_SEC,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Write `url` to `dest`. Callers pass a staging path, not the user file."""
    if not is_allowed_asset_url(url):
        raise ValueError("refusing to download from an unexpected URL")
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "RallyClip"})
    try:
        with urlopen(request, timeout=timeout) as response, dest.open("wb") as out:
            while True:
                _raise_if_cancelled(cancel_event)
                chunk = response.read(_HASH_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise


def download_and_open_latest_dmg(
    latest: dict[str, Any],
    *,
    cancel_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    dmg_url = str(latest.get("dmg_url") or "")
    sha_url = str(latest.get("sha256_url") or "")
    if not dmg_url or not sha_url:
        raise ValueError("Latest GitHub release has no macOS arm64 DMG and checksum.")
    dmg_name = Path(urlparse(dmg_url).path).name
    if not DMG_NAME_RE.fullmatch(dmg_name):
        raise ValueError(f"unexpected DMG name: {dmg_name}")
    dest = downloads_dir() / dmg_name
    sha_dest = dest.with_name(dest.name + ".sha256")
    token = uuid.uuid4().hex[:8]
    staging = dest.with_name(f"{dest.name}.{token}.partial")
    sha_staging = sha_dest.with_name(f"{sha_dest.name}.{token}.partial")
    try:
        _raise_if_cancelled(cancel_event)
        _download_url(sha_url, sha_staging, cancel_event=cancel_event)
        expected = parse_dmg_sha256_sidecar(sha_staging.read_text(encoding="utf-8"), dmg_name)
        _raise_if_cancelled(cancel_event)
        _download_url(dmg_url, staging, cancel_event=cancel_event)
        _raise_if_cancelled(cancel_event)
        if sha256_file(staging) != expected:
            raise ValueError("DMG checksum mismatch; download discarded.")
        sha_staging.replace(sha_dest)
        staging.replace(dest)
    except Exception:
        staging.unlink(missing_ok=True)
        sha_staging.unlink(missing_ok=True)
        raise
    _raise_if_cancelled(cancel_event)
    open_downloaded_dmg(dest)
    logging.info("Opened downloaded update DMG %s", dest)
    return {
        "opened": True,
        "path": str(dest),
        "release_url": latest.get("release_url") or GITHUB_RELEASES_URL,
    }
