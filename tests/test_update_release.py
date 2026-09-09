"""GitHub Latest DMG asset picking and checksum-verified download."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from gui.update_release import (
    ASSET_DOWNLOAD_PREFIX,
    UpdateDownloadCancelled,
    begin_update_download,
    download_and_open_latest_dmg,
    install_channel,
    parse_dmg_sha256_sidecar,
    parse_latest_release,
    pick_macos_arm64_dmg_assets,
    request_update_cancel,
    select_latest_app_release,
)


def _asset(name: str, *, tag: str = "v0.5.1") -> dict:
    return {
        "name": name,
        "browser_download_url": f"{ASSET_DOWNLOAD_PREFIX}{tag}/{name}",
    }


def test_pick_macos_arm64_dmg_skips_artifact_zip_and_intel():
    dmg, sha = pick_macos_arm64_dmg_assets(
        [
            _asset("artifact-rallyclip_v0.5.0.zip", tag="artifact-rallyclip_v0.5.0"),
            _asset("RallyClip-0.5.1-macOS-x86_64.dmg"),
            _asset("RallyClip-0.5.1-macOS-arm64.dmg"),
            _asset("RallyClip-0.5.1-macOS-arm64.dmg.sha256"),
        ]
    )
    assert dmg and dmg.endswith("RallyClip-0.5.1-macOS-arm64.dmg")
    assert sha and sha.endswith("RallyClip-0.5.1-macOS-arm64.dmg.sha256")


def test_select_latest_app_release_skips_newer_artifact():
    artifact = {
        "tag_name": "artifact-rallyclip_v0.5.1",
        "draft": False,
        "prerelease": False,
        "published_at": "2026-09-10T00:00:00Z",
        "html_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/artifact-rallyclip_v0.5.1",
        "assets": [_asset("rallyclip_v0.5.1.zip", tag="artifact-rallyclip_v0.5.1")],
    }
    app = {
        "tag_name": "v0.5.1",
        "draft": False,
        "prerelease": False,
        "published_at": "2026-09-09T00:00:00Z",
        "html_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/v0.5.1",
        "name": "RallyClip 0.5.1",
        "assets": [
            _asset("RallyClip-0.5.1-macOS-arm64.dmg"),
            _asset("RallyClip-0.5.1-macOS-arm64.dmg.sha256"),
        ],
    }
    parsed = select_latest_app_release([artifact, app])
    assert parsed is not None
    assert parsed["latest_tag"] == "v0.5.1"
    assert parsed["dmg_url"].endswith("RallyClip-0.5.1-macOS-arm64.dmg")


def test_select_latest_app_release_skips_prerelease_and_drafts():
    parsed = select_latest_app_release(
        [
            {
                "tag_name": "v0.5.2",
                "draft": False,
                "prerelease": True,
                "published_at": "2026-09-11T00:00:00Z",
                "assets": [
                    _asset("RallyClip-0.5.2-macOS-arm64.dmg", tag="v0.5.2"),
                    _asset("RallyClip-0.5.2-macOS-arm64.dmg.sha256", tag="v0.5.2"),
                ],
            },
            {
                "tag_name": "v0.5.1",
                "draft": True,
                "prerelease": False,
                "published_at": "2026-09-10T00:00:00Z",
                "assets": [
                    _asset("RallyClip-0.5.1-macOS-arm64.dmg"),
                    _asset("RallyClip-0.5.1-macOS-arm64.dmg.sha256"),
                ],
            },
            {
                "tag_name": "v0.5.0",
                "draft": False,
                "prerelease": False,
                "published_at": "2026-09-08T00:00:00Z",
                "html_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/v0.5.0",
                "assets": [
                    _asset("RallyClip-0.5.0-macOS-arm64.dmg", tag="v0.5.0"),
                    _asset("RallyClip-0.5.0-macOS-arm64.dmg.sha256", tag="v0.5.0"),
                ],
            },
        ]
    )
    assert parsed is not None
    assert parsed["latest_tag"] == "v0.5.0"


def test_parse_latest_release_includes_dmg_urls():
    parsed = parse_latest_release(
        {
            "tag_name": "v0.5.1",
            "html_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/v0.5.1",
            "name": "RallyClip 0.5.1",
            "assets": [
                _asset("RallyClip-0.5.1-macOS-arm64.dmg"),
                _asset("RallyClip-0.5.1-macOS-arm64.dmg.sha256"),
            ],
        }
    )
    assert parsed["latest_version"] == "0.5.1"
    assert parsed["dmg_url"].endswith("RallyClip-0.5.1-macOS-arm64.dmg")
    assert parsed["sha256_url"].endswith(".dmg.sha256")


def test_parse_dmg_sha256_sidecar_matches_name():
    text = ("a" * 64) + "  other.dmg\n" + ("b" * 64) + "  RallyClip-0.5.1-macOS-arm64.dmg\n"
    assert parse_dmg_sha256_sidecar(text, "RallyClip-0.5.1-macOS-arm64.dmg") == "b" * 64


def test_parse_dmg_sha256_sidecar_requires_named_digest():
    dmg_name = "RallyClip-0.5.1-macOS-arm64.dmg"
    with pytest.raises(ValueError, match="did not contain"):
        parse_dmg_sha256_sidecar("a" * 64 + "\n", dmg_name)
    with pytest.raises(ValueError, match="did not contain"):
        parse_dmg_sha256_sidecar(("a" * 64) + "  other.dmg\n", dmg_name)


def test_begin_update_download_cancels_previous():
    _gen1, first = begin_update_download()
    _gen2, second = begin_update_download()
    assert first is not second
    assert first.is_set()
    assert not second.is_set()
    request_update_cancel()
    assert second.is_set()


def test_install_channel_follows_frozen_flag(monkeypatch):
    monkeypatch.setattr("gui.update_release.sys.frozen", False, raising=False)
    assert install_channel() == "source"
    monkeypatch.setattr("gui.update_release.sys.frozen", True, raising=False)
    assert install_channel() == "dmg"


def test_download_opens_dmg_when_hash_matches(tmp_path, monkeypatch):
    dmg_name = "RallyClip-0.5.1-macOS-arm64.dmg"
    opened: list[Path] = []
    monkeypatch.setattr("gui.update_release.downloads_dir", lambda: tmp_path)

    def fake_download(
        url: str, dest: Path, *, timeout: float = 300, cancel_event=None
    ) -> None:
        body = b"dmg-bytes"
        if url.endswith(".sha256"):
            dest.write_text(
                f"{hashlib.sha256(body).hexdigest()}  {dmg_name}\n",
                encoding="utf-8",
            )
            return
        dest.write_bytes(body)

    monkeypatch.setattr("gui.update_release._download_url", fake_download)
    monkeypatch.setattr("gui.update_release.open_downloaded_dmg", opened.append)

    latest = parse_latest_release(
        {
            "tag_name": "v0.5.1",
            "html_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/v0.5.1",
            "assets": [_asset(dmg_name), _asset(f"{dmg_name}.sha256")],
        }
    )
    result = download_and_open_latest_dmg(latest)
    assert result["opened"] is True
    assert opened == [tmp_path / dmg_name]
    assert (tmp_path / dmg_name).is_file()


def test_download_discards_dmg_on_hash_mismatch(tmp_path, monkeypatch):
    dmg_name = "RallyClip-0.5.1-macOS-arm64.dmg"
    opened: list[Path] = []
    monkeypatch.setattr("gui.update_release.downloads_dir", lambda: tmp_path)

    def fake_download(
        url: str, dest: Path, *, timeout: float = 300, cancel_event=None
    ) -> None:
        if url.endswith(".sha256"):
            dest.write_text(f"{'0' * 64}  {dmg_name}\n", encoding="utf-8")
            return
        dest.write_bytes(b"tampered")

    monkeypatch.setattr("gui.update_release._download_url", fake_download)
    monkeypatch.setattr("gui.update_release.open_downloaded_dmg", opened.append)

    latest = parse_latest_release(
        {
            "tag_name": "v0.5.1",
            "html_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/v0.5.1",
            "assets": [_asset(dmg_name), _asset(f"{dmg_name}.sha256")],
        }
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_and_open_latest_dmg(latest)
    assert opened == []
    assert not (tmp_path / dmg_name).exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_download_preserves_existing_dmg_on_mismatch(tmp_path, monkeypatch):
    dmg_name = "RallyClip-0.5.1-macOS-arm64.dmg"
    dest = tmp_path / dmg_name
    dest.write_bytes(b"keep-me")
    opened: list[Path] = []
    monkeypatch.setattr("gui.update_release.downloads_dir", lambda: tmp_path)

    def fake_download(
        url: str, dest_path: Path, *, timeout: float = 300, cancel_event=None
    ) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".sha256"):
            dest_path.write_text(f"{'0' * 64}  {dmg_name}\n", encoding="utf-8")
            return
        dest_path.write_bytes(b"tampered")

    monkeypatch.setattr("gui.update_release._download_url", fake_download)
    monkeypatch.setattr("gui.update_release.open_downloaded_dmg", opened.append)

    latest = parse_latest_release(
        {
            "tag_name": "v0.5.1",
            "html_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/v0.5.1",
            "assets": [_asset(dmg_name), _asset(f"{dmg_name}.sha256")],
        }
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_and_open_latest_dmg(latest)
    assert opened == []
    assert dest.read_bytes() == b"keep-me"
    assert list(tmp_path.glob("*.partial")) == []


def test_download_does_not_open_when_cancelled(tmp_path, monkeypatch):
    dmg_name = "RallyClip-0.5.1-macOS-arm64.dmg"
    dest = tmp_path / dmg_name
    dest.write_bytes(b"keep-me")
    opened: list[Path] = []
    cancel = threading.Event()
    monkeypatch.setattr("gui.update_release.downloads_dir", lambda: tmp_path)

    def fake_download(
        url: str, dest_path: Path, *, timeout: float = 300, cancel_event=None
    ) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        body = b"dmg-bytes"
        if url.endswith(".sha256"):
            dest_path.write_text(
                f"{hashlib.sha256(body).hexdigest()}  {dmg_name}\n",
                encoding="utf-8",
            )
            if cancel_event is not None:
                cancel_event.set()
            return
        dest_path.write_bytes(body)

    monkeypatch.setattr("gui.update_release._download_url", fake_download)
    monkeypatch.setattr("gui.update_release.open_downloaded_dmg", opened.append)

    latest = parse_latest_release(
        {
            "tag_name": "v0.5.1",
            "html_url": "https://github.com/iroblesrazzaq/RallyClip/releases/tag/v0.5.1",
            "assets": [_asset(dmg_name), _asset(f"{dmg_name}.sha256")],
        }
    )
    with pytest.raises(UpdateDownloadCancelled):
        download_and_open_latest_dmg(latest, cancel_event=cancel)
    assert opened == []
    assert dest.read_bytes() == b"keep-me"
    assert list(tmp_path.glob("*.partial")) == []


def test_download_url_stops_on_cancel(tmp_path, monkeypatch):
    from gui.update_release import _download_url

    cancel = threading.Event()
    reads = {"n": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            reads["n"] += 1
            if reads["n"] >= 2:
                cancel.set()
            return b"x" * min(size, 64)

    monkeypatch.setattr("gui.update_release.urlopen", lambda *args, **kwargs: _Response())
    dest = tmp_path / "RallyClip-0.5.1-macOS-arm64.dmg.partial"
    url = f"{ASSET_DOWNLOAD_PREFIX}v0.5.1/RallyClip-0.5.1-macOS-arm64.dmg"
    with pytest.raises(UpdateDownloadCancelled):
        _download_url(url, dest, cancel_event=cancel)
    assert not dest.exists()
