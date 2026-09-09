"""GitHub Latest DMG asset picking and checksum-verified download."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gui.update_release import (
    ASSET_DOWNLOAD_PREFIX,
    download_and_open_latest_dmg,
    install_channel,
    parse_dmg_sha256_sidecar,
    parse_latest_release,
    pick_macos_arm64_dmg_assets,
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


def test_install_channel_follows_frozen_flag(monkeypatch):
    monkeypatch.setattr("gui.update_release.sys.frozen", False, raising=False)
    assert install_channel() == "source"
    monkeypatch.setattr("gui.update_release.sys.frozen", True, raising=False)
    assert install_channel() == "dmg"


def test_download_opens_dmg_when_hash_matches(tmp_path, monkeypatch):
    dmg_name = "RallyClip-0.5.1-macOS-arm64.dmg"
    opened: list[Path] = []
    monkeypatch.setattr("gui.update_release.downloads_dir", lambda: tmp_path)

    def fake_download(url: str, dest: Path, *, timeout: float = 300) -> None:
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

    def fake_download(url: str, dest: Path, *, timeout: float = 300) -> None:
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
