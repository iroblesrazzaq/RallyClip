"""L1 backend-journey e2e tests — the release gate.

Boots the *real* Flask backend (the one the desktop webview drives) on a free
localhost port and walks the full user journey over HTTP, no mocks:

    upload-and-start  ->  poll progress to completion  ->  saved library item

The pipeline (YOLO pose -> court mask -> features -> ONNX inference -> decode)
runs for real on a synthetic clip, so assertions are structural: the job
completes, progress advances monotonically, bad inputs are rejected, and a
points-free synthetic clip saves no library item. The library API (list /
thumbnail / video / csv / delete) is exercised with fabricated items so it runs
in CI without real footage; the real save->export path is covered by the golden
tests. Segment *values* are only asserted on the golden clip.

Marked ``e2e``+``slow``; deselected from the default run. Run explicitly:

    pytest -m e2e tests/test_gui_e2e.py

Skips cleanly when runtime deps or the shipped v0.3.1 ONNX artifact are absent.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODEL_ONNX = REPO_ROOT / "models" / "rallyclip_v0.3.1" / "model.onnx"

pytest.importorskip("requests")
pytest.importorskip("cv2")
pytest.importorskip("ultralytics")
pytest.importorskip("onnxruntime")
pytest.importorskip("av")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from helpers.e2e_backend import BackendClient, running_backend  # noqa: E402
from helpers.golden_metrics import score_six_bin  # noqa: E402

# Golden clip fixture: a short window cut from a real training match (private,
# gitignored under data/). Present locally where the training data lives; absent
# in CI, where these tests self-skip. Generate with scripts/make_e2e_clip.py.
GOLDEN_DIR = Path(os.environ.get("RALLYCLIP_E2E_CLIP_DIR") or (REPO_ROOT / "data" / "e2e" / "aditi_5pts"))
GOLDEN_CLIP = GOLDEN_DIR / "clip.mp4"
GOLDEN_JSON = GOLDEN_DIR / "golden.json"
_HAS_GOLDEN = GOLDEN_CLIP.is_file() and GOLDEN_JSON.is_file()
golden = pytest.mark.skipif(
    not _HAS_GOLDEN,
    reason=f"golden clip absent ({GOLDEN_DIR}); generate with scripts/make_e2e_clip.py",
)
GOLDEN_TIMEOUT = 600.0  # 130s clip -> full pose extraction; generous for slow CI

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        not MODEL_ONNX.exists(),
        reason="shipped v0.3.1 ONNX artifact missing (models/rallyclip_v0.3.1/model.onnx)",
    ),
]

CLIP_SECONDS = 30.0  # > seq_len/fps (100/5 = 20s) so windowed inference gets a full window
JOB_TIMEOUT = 360.0  # generous: real CPU pose extraction on a synthetic clip


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def synthetic_clip(tmp_path_factory) -> Path:
    """A short synthetic MP4 (reuses the CI smoke-clip generator)."""
    from make_smoke_clip import make_clip

    clip = tmp_path_factory.mktemp("e2e_clip") / "smoke.mp4"
    make_clip(clip, duration_s=CLIP_SECONDS)
    assert clip.exists() and clip.stat().st_size > 0
    return clip


@pytest.fixture(scope="module")
def backend(tmp_path_factory, monkeypatch_module) -> BackendClient:
    """Real backend on a free port with isolated jobs/output/csv/library dirs."""
    root = tmp_path_factory.mktemp("e2e_backend")
    monkeypatch_module.setenv("RALLYCLIP_KEEP_JOBS", "1")  # don't sweep/delete outputs mid-run
    with running_backend(root / "jobs", root / "output", root / "csv", root / "library") as client:
        yield client


@pytest.fixture
def library_dir(backend: BackendClient) -> Path:
    """The temp library dir the running backend writes to (redirected by the
    backend fixture). Function-scoped: reads the live global after redirect."""
    from gui import app as gui_app

    return Path(gui_app.LIBRARY_DIR)


def _fabricate_library_item(library_dir: Path, *, n_segments: int = 3, with_csv: bool = True, with_thumb: bool = True) -> str:
    """Drop a fake library item on disk (no pipeline) so the library API can be
    tested in CI without real footage. Returns the item id."""
    import uuid

    item_id = f"20990101-000000-{uuid.uuid4().hex[:6]}"
    item_dir = library_dir / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "source.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake-video-bytes")
    if with_csv:
        (item_dir / "segments.csv").write_text("start_time,end_time\n1.000,3.500\n", encoding="utf-8")
    if with_thumb:
        (item_dir / "thumb.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    meta = {
        "id": item_id,
        "name": "Fabricated Match",
        "source_name": "match.mp4",
        "created": "2099-01-01T00:00:00",
        "created_ts": time.time(),
        "duration_s": 12.5,
        "n_segments": n_segments,
    }
    (item_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return item_id


@pytest.fixture(scope="module")
def monkeypatch_module():
    """A module-scoped monkeypatch (pytest's built-in one is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        yield mp
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def default_job(backend: BackendClient, synthetic_clip: Path):
    """A real run with default thresholds — the realistic happy path. A synthetic
    clip yields ~zero points, so no library item is saved (nothing worth keeping)
    and progress reports library_id=None."""
    resp = backend.start_job(synthetic_clip, {"write_csv": True, "segment_video": True})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    body, snapshots = backend.wait_for(job_id, timeout=JOB_TIMEOUT)
    return job_id, body, snapshots


# --------------------------------------------------------------------------- #
# Fast endpoint contract tests (no pipeline run)
# --------------------------------------------------------------------------- #
def test_health_ok(backend: BackendClient):
    resp = backend.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_config_defaults_contract(backend: BackendClient):
    resp = backend.get("/api/config/defaults")
    assert resp.status_code == 200
    payload = resp.json()
    defaults = payload["defaults"]
    assert defaults["fps"] == 5.0
    assert defaults["feature_set"] == "v1"
    assert payload["yolo_model"] == "yolov8n-pose.pt"
    assert defaults["yolo_size"] == "nano"
    assert "available_devices" in payload
    assert "auto_device" in payload
    # Server-internal absolute paths must not leak to the browser payload.
    for leaked in ("model_path", "scaler_path", "artifact_dir", "yolo_weights"):
        assert leaked not in defaults


def test_upload_rejects_non_video(backend: BackendClient, tmp_path: Path):
    # Policy: accept any decodable container, reject by content (not extension).
    # A non-video file is rejected cleanly at upload, before any job spawns.
    bad = tmp_path / "notes.txt"
    bad.write_text("not a video", encoding="utf-8")
    with open(bad, "rb") as fh:
        resp = backend.post(
            "/api/upload-and-start",
            files={"video": ("notes.txt", fh, "text/plain")},
            data={"config": "{}"},
        )
    assert resp.status_code == 400
    assert "could not be opened" in resp.json()["error"].lower()


def test_upload_rejects_missing_field(backend: BackendClient):
    resp = backend.post("/api/upload-and-start", data={"config": "{}"})
    assert resp.status_code == 400
    assert "video" in resp.json()["error"].lower()


def test_progress_unknown_job_404(backend: BackendClient):
    resp = backend.progress("00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_library_unknown_item_404(backend: BackendClient):
    missing = "does-not-exist"
    assert backend.get(f"/api/library/{missing}/video").status_code == 404
    assert backend.get(f"/api/library/{missing}/csv").status_code == 404
    assert backend.get(f"/api/library/{missing}/thumbnail").status_code == 404


def test_library_id_cannot_escape(backend: BackendClient, library_dir: Path):
    # Defense in depth: even if a traversal id reached the handler, the resolver
    # rejects anything escaping the library dir. (Werkzeug also normalizes the
    # URL before routing, so this can't be hit over HTTP — hence the direct test.)
    from gui import app as gui_app

    with pytest.raises(ValueError):
        gui_app._library_item_dir("../../etc/passwd")
    assert gui_app._library_item_dir("ok-123").parent == library_dir


# --------------------------------------------------------------------------- #
# Library API tests (fabricated items — no pipeline, CI-safe)
# --------------------------------------------------------------------------- #
def test_library_lists_fabricated_item(backend: BackendClient, library_dir: Path):
    item_id = _fabricate_library_item(library_dir, n_segments=4)
    resp = backend.get("/api/library")
    assert resp.status_code == 200
    items = resp.json()["items"]
    item = next((i for i in items if i["id"] == item_id), None)
    assert item is not None, "fabricated item not listed"
    assert item["n_segments"] == 4
    assert item["has_csv"] and item["has_thumbnail"]


def test_library_item_downloads(backend: BackendClient, library_dir: Path):
    item_id = _fabricate_library_item(library_dir)
    thumb = backend.get(f"/api/library/{item_id}/thumbnail")
    assert thumb.status_code == 200 and thumb.content
    video = backend.get(f"/api/library/{item_id}/preview")
    assert video.status_code == 200 and video.content
    segments = backend.get(f"/api/library/{item_id}/segments")
    assert segments.status_code == 200
    assert segments.json()["segments"] == [{"start": 1.0, "end": 3.5}]
    csv = backend.get(f"/api/library/{item_id}/csv")
    assert csv.status_code == 200
    assert csv.text.splitlines()[0].strip() == "start_time,end_time"


def test_library_delete_removes_item(backend: BackendClient, library_dir: Path):
    item_id = _fabricate_library_item(library_dir)
    assert (library_dir / item_id).exists()
    resp = backend.delete(f"/api/library/{item_id}")
    assert resp.status_code == 200
    assert not (library_dir / item_id).exists()
    assert backend.get(f"/api/library/{item_id}/video").status_code == 404
    items = backend.get("/api/library").json()["items"]
    assert all(i["id"] != item_id for i in items)


def test_library_item_without_csv_omits_flag(backend: BackendClient, library_dir: Path):
    item_id = _fabricate_library_item(library_dir, with_csv=False)
    items = backend.get("/api/library").json()["items"]
    item = next(i for i in items if i["id"] == item_id)
    assert item["has_csv"] is False
    assert backend.get(f"/api/library/{item_id}/csv").status_code == 404


# --------------------------------------------------------------------------- #
# Full-journey tests (real pipeline)
# --------------------------------------------------------------------------- #
def test_default_job_completes(default_job):
    _job_id, body, _snaps = default_job
    assert body["status"] == "completed", body.get("error")
    assert body.get("error") is None
    for step in ("pose", "preprocess", "feature", "inference", "output"):
        assert body["steps"][step]["status"] == "completed"
        assert body["steps"][step]["progress"] == 100


def test_progress_is_monotonic(default_job):
    """Each step's progress never decreases across polls, and the job terminates."""
    _job_id, body, snaps = default_job
    assert snaps, "expected at least one progress snapshot"
    for step in ("pose", "preprocess", "feature", "inference", "output"):
        seq = [s["steps"][step]["progress"] for s in snaps]
        for prev, cur in zip(seq, seq[1:]):
            assert cur >= prev, f"{step} progress went backwards: {seq}"
    assert body["status"] == "completed"


def test_default_job_saves_no_library_item(default_job):
    """A synthetic clip has no real points -> nothing worth saving. The job still
    completes, but produces no library item (library_id is None)."""
    _job_id, body, _snaps = default_job
    assert body["status"] == "completed"
    assert body.get("library_id") is None


def test_cancel_stops_job(backend: BackendClient, synthetic_clip: Path):
    resp = backend.start_job(synthetic_clip, {"write_csv": True, "segment_video": False})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    # Cancel almost immediately; the long pose stage gives ample time to land.
    time.sleep(0.3)
    cancel = backend.post(f"/api/cancel/{job_id}")
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"

    body, snaps = backend.wait_for(job_id, timeout=JOB_TIMEOUT)
    assert body["status"] == "cancelled"
    assert all(s["status"] != "completed" for s in snaps), "job must not reach completed after cancel"


# --------------------------------------------------------------------------- #
# Golden-clip tests (real match footage, self-skip when absent)
# --------------------------------------------------------------------------- #
def _parse_csv_segments(text: str) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for line in text.splitlines()[1:]:  # skip header
        parts = [p for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            segments.append((float(parts[0]), float(parts[1])))
    return segments


@pytest.fixture(scope="module")
def golden_points() -> list[tuple[float, float]]:
    data = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    return [tuple(p) for p in data["points"]]


@pytest.fixture(scope="module")
def golden_job(backend: BackendClient):
    """A real run on the golden clip -> a saved library item with detected
    segments. Returns (library_id, body, detected_segments)."""
    resp = backend.start_job(GOLDEN_CLIP, {"write_csv": True, "segment_video": True})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    body, _snaps = backend.wait_for(job_id, timeout=GOLDEN_TIMEOUT)
    lib_id = body.get("library_id")
    detected: list[tuple[float, float]] = []
    if lib_id:
        csv_resp = backend.get(f"/api/library/{lib_id}/csv")
        assert csv_resp.status_code == 200, csv_resp.text
        detected = _parse_csv_segments(csv_resp.text)
    return lib_id, body, detected


@golden
def test_golden_job_saves_library_item(backend: BackendClient, golden_job):
    """Real points -> the match is saved to the library and shows up in the list."""
    lib_id, body, _detected = golden_job
    assert body["status"] == "completed", body.get("error")
    assert lib_id, "real points should produce a library item"
    items = backend.get("/api/library").json()["items"]
    item = next((i for i in items if i["id"] == lib_id), None)
    assert item is not None, "saved match not listed in the library"
    assert item["n_segments"] >= 1
    assert item["has_csv"] and item["has_thumbnail"]


@golden
def test_golden_model_detects_labeled_points(golden_job, golden_points):
    """The pipeline should recover the labeled points with few false positives.
    These are training videos, so the model is optimistic — the floor is a loose
    regression bar (1-point margin), not a generalization measure."""
    _lib_id, _body, detected = golden_job
    result = score_six_bin(detected, golden_points)
    n = len(golden_points)
    assert result["n_pred"] >= n - 1, f"model under-fired: {result}"
    assert result["acceptable_frac"] >= 0.8, f"too few acceptable points: {result}"
    assert result["fp"] <= 1, f"too many false positives: {result}"
    assert result["fn"] <= 1, f"too many misses: {result}"


@golden
def test_golden_library_video_is_downloadable(backend: BackendClient, golden_job, tmp_path: Path):
    """The saved match exports a real, openable MP4 — the full ingest -> detect ->
    cut -> save -> export path synthetic clips can't reach."""
    import av

    lib_id, _body, detected = golden_job
    assert lib_id
    resp = backend.get(f"/api/library/{lib_id}/video")
    assert resp.status_code == 200, resp.text
    assert resp.content, "empty video payload"
    out = tmp_path / "golden_segmented.mp4"
    out.write_bytes(resp.content)
    with av.open(str(out)) as container:
        assert container.streams.video, "downloaded file has no video stream"
        duration = float(container.duration) / 1e6 if container.duration else 0.0
    expected = sum(e - s for s, e in detected)
    assert duration > 0
    assert duration <= expected + 5.0, f"segmented video longer than detected points: {duration:.1f} vs ~{expected:.1f}"
