import re
from types import SimpleNamespace

from flask import Flask
import pytest

from gui.native_backend import register_native_routes

TOKEN = "test-native-token-" * 3


@pytest.fixture
def service(tmp_path):
    app = Flask(__name__)
    calls = []
    def item_dir(item_id):
        if not re.fullmatch(r"[a-z0-9-]+", item_id):
            raise ValueError("Invalid id")
        return tmp_path / item_id
    fake = SimpleNamespace(
        app=app,
        _natural_filename_key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)],
        _normalize_config=lambda cfg: cfg,
        _start_folder_analysis_job=lambda sources, cfg: calls.append((sources, cfg)) or "job-1",
        _library_item_dir=item_dir,
        _resolve_library_source=lambda item_id: tmp_path / item_id / "source.mp4",
        _estimate_duration_seconds=lambda p: 30,
    )
    register_native_routes(fake, TOKEN)
    return app.test_client(), calls, tmp_path


def test_native_service_requires_token_on_reads_and_writes(service):
    client, _, _ = service
    for method, url in [("GET", "/api/native/library/match/sources"), ("POST", "/api/native/import")]:
        assert client.open(url, method=method).status_code == 401
        assert client.open(url, method=method, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_native_service_refuses_browser_and_rebinding_requests(service):
    client, _, _ = service
    headers = {"Authorization": f"Bearer {TOKEN}", "Origin": "http://127.0.0.1"}
    assert client.post("/api/native/import", headers=headers, base_url="http://127.0.0.1").status_code == 403
    headers.pop("Origin")
    assert client.post("/api/native/import", headers=headers, base_url="http://evil.test").status_code == 403


def test_native_import_orders_clips_and_preserves_name(service):
    client, calls, root = service
    files = [root / name for name in ("clip10.mp4", "clip2.mp4", "clip1.mp4")]
    for path in files:
        path.touch()
    result = client.post("/api/native/import", json={"paths": [str(p) for p in files], "name": "My match"},
                         headers={"Authorization": f"Bearer {TOKEN}"}, base_url="http://127.0.0.1")
    assert result.status_code == 202
    assert [p.name for p in calls[0][0]] == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]
    assert calls[0][1]["output_name"] == "My match"


@pytest.mark.parametrize("paths", [[], [None], ["/missing.mp4"], "not-a-list"])
def test_native_import_rejects_invalid_sources(service, paths):
    client, calls, _ = service
    result = client.post("/api/native/import", json={"paths": paths}, headers={"Authorization": f"Bearer {TOKEN}"}, base_url="http://127.0.0.1")
    assert result.status_code == 400
    assert not calls


def test_missing_originals_are_reported_before_playback(service):
    client, _, root = service
    item = root / "match"
    item.mkdir()
    (item / "sources.json").write_text('{"sources":[{"path":"/missing-original.mp4"}]}')
    result = client.get("/api/native/library/match/sources", headers={"Authorization": f"Bearer {TOKEN}"}, base_url="http://127.0.0.1")
    assert result.status_code == 409
    assert "Reconnect" in result.json["error"]
