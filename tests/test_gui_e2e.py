"""L1 backend-journey e2e tests — the release gate.

Boots the *real* Flask backend (the one the desktop webview drives) on a free
localhost port and walks the full user journey over HTTP, no mocks:

    upload-and-start  ->  poll progress to completion  ->  download CSV / video

The pipeline (YOLO pose -> court mask -> features -> ONNX inference -> decode)
runs for real on a synthetic clip, so assertions are structural: the job
completes, progress advances monotonically, the CSV has the right header, the
video output path behaves, bad inputs are rejected. Segment *values* are not
asserted — a synthetic clip contains no real tennis points.

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
    """Real backend on a free port with isolated jobs/output/csv dirs."""
    root = tmp_path_factory.mktemp("e2e_backend")
    monkeypatch_module.setenv("RALLYCLIP_KEEP_JOBS", "1")  # don't sweep/delete outputs mid-run
    with running_backend(root / "jobs", root / "output", root / "csv") as client:
        yield client


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
    """A real run with default thresholds, both outputs enabled — the realistic
    happy path. A synthetic clip yields ~zero points, so the CSV is header-only
    and no video file is produced; both download endpoints behave accordingly."""
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
    assert payload["yolo_sizes"]  # non-empty list of sizes
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


def test_download_unknown_job_404(backend: BackendClient):
    missing = "00000000-0000-0000-0000-000000000000"
    assert backend.get(f"/api/download/csv/{missing}").status_code == 404
    assert backend.get(f"/api/download/video/{missing}").status_code == 404


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


def test_csv_download_has_header(backend: BackendClient, default_job):
    job_id, _body, _snaps = default_job
    resp = backend.get(f"/api/download/csv/{job_id}")
    assert resp.status_code == 200, resp.text
    first_line = resp.text.splitlines()[0].strip()
    assert first_line == "start_time,end_time"


def test_video_download_reflects_segments(backend: BackendClient, default_job, tmp_path: Path):
    """Video download mirrors what the pipeline produced: a real, openable MP4 when
    points were found, or an explicit 404 (no server error) when none were — which
    is the synthetic-clip case. Forward-compatible with a real-match clip fixture.
    The video *encode* path itself is covered directly by tests/test_segment.py."""
    job_id, _body, _snaps = default_job
    resp = backend.get(f"/api/download/video/{job_id}")
    if resp.status_code == 200:
        import av

        assert resp.content, "empty video payload"
        out = tmp_path / "downloaded_segmented.mp4"
        out.write_bytes(resp.content)
        with av.open(str(out)) as container:
            assert container.streams.video, "downloaded file has no video stream"
    else:
        assert resp.status_code == 404
        assert "not available" in resp.json()["error"].lower()


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
    """A real run on the golden clip (real points -> real CSV + segmented video)."""
    resp = backend.start_job(GOLDEN_CLIP, {"write_csv": True, "segment_video": True})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    body, _snaps = backend.wait_for(job_id, timeout=GOLDEN_TIMEOUT)
    csv_resp = backend.get(f"/api/download/csv/{job_id}")
    assert csv_resp.status_code == 200, csv_resp.text
    detected = _parse_csv_segments(csv_resp.text)
    return job_id, body, detected


@golden
def test_golden_job_completes(golden_job):
    _job_id, body, _detected = golden_job
    assert body["status"] == "completed", body.get("error")


@golden
def test_golden_model_detects_labeled_points(golden_job, golden_points):
    """The pipeline should recover the labeled points with few false positives.
    These are training videos, so the model is optimistic — the floor is a loose
    regression bar (1-point margin), not a generalization measure."""
    _job_id, _body, detected = golden_job
    result = score_six_bin(detected, golden_points)
    n = len(golden_points)
    assert result["n_pred"] >= n - 1, f"model under-fired: {result}"
    assert result["acceptable_frac"] >= 0.8, f"too few acceptable points: {result}"
    assert result["fp"] <= 1, f"too many false positives: {result}"
    assert result["fn"] <= 1, f"too many misses: {result}"


@golden
def test_golden_video_is_downloadable(backend: BackendClient, golden_job, tmp_path: Path):
    """Real points were found, so a real segmented MP4 is produced and served —
    the full ingest -> detect -> cut -> download path synthetic clips can't reach."""
    import av

    job_id, _body, detected = golden_job
    resp = backend.get(f"/api/download/video/{job_id}")
    assert resp.status_code == 200, resp.text
    assert resp.content, "empty video payload"
    out = tmp_path / "golden_segmented.mp4"
    out.write_bytes(resp.content)
    with av.open(str(out)) as container:
        assert container.streams.video, "downloaded file has no video stream"
        duration = float(container.duration) / 1e6 if container.duration else 0.0
    # The cut concatenates the detected points; its length should be in the
    # ballpark of their summed durations (loose: re-encode/keyframe slack).
    expected = sum(e - s for s, e in detected)
    assert duration > 0
    assert duration <= expected + 5.0, f"segmented video longer than detected points: {duration:.1f} vs ~{expected:.1f}"
