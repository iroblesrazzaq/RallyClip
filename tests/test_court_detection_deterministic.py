"""Court-detection regression -- Flight 1: DETERMINISTIC (post-homography).

Feeds the frozen clean frames (already homographied; no players occluding the
lines) straight into the detection pipeline, bypassing ``extract_clean_frame``.
Everything downstream (Canny/Hough/morphology/geometry) is deterministic for
identical input pixels, so this flight is a tight tripwire: any new heuristic
that moves a mask on the known-good courts trips it. Tolerance is plain
numerical error (IoU >= 1 - 1e-3).

Regenerate fixtures with ``scripts/court_fixtures_gen.py``.
"""
from __future__ import annotations

import pytest

from helpers.court_fixtures import FIX_DIR, fixture_ids, iou

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

DET_TOL = 1e-3  # numerical tolerance: IoU must be >= 1 - DET_TOL
FIXTURE_IDS = fixture_ids()


def _detector(monkeypatch, frame):
    import preprocessing.court_detector_impl as cdi
    monkeypatch.setattr(cdi, "YOLO_AVAILABLE", False)  # no YOLO load; we feed the frame
    det = cdi.CourtDetector(yolo_model_path="unused", conf=0.25)  # YOLO bypassed; conf unused here
    monkeypatch.setattr(det, "extract_clean_frame", lambda video_path, target_time=60: frame.copy())
    return det


def test_fixture_set_present():
    assert len(FIXTURE_IDS) >= 11, f"expected >=11 court fixtures, got {len(FIXTURE_IDS)}"


def test_draw_court_lines_handles_missing_baseline(monkeypatch):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    det = _detector(monkeypatch, frame)

    rendered = det.draw_court_lines(
        frame,
        baseline=None,
        left_doubles_sideline=[[20, 110, 45, 10]],
        right_doubles_sideline=[[140, 110, 115, 10]],
    )

    assert rendered.shape == frame.shape


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_detection_deterministic(monkeypatch, fixture_id):
    frame = cv2.imread(str(FIX_DIR / "frames" / f"{fixture_id}.png"))
    golden = cv2.imread(str(FIX_DIR / "masks" / f"{fixture_id}.png"), cv2.IMREAD_GRAYSCALE)
    assert frame is not None and golden is not None, f"missing fixture files for {fixture_id}"
    # OpenCV 4.13 on some platforms returns (H, W, 1) for grayscale reads.
    golden = golden.reshape(golden.shape[0], golden.shape[1])

    det = _detector(monkeypatch, frame)
    out_mask, _, meta = det.process_video("dummy.mp4", target_time=60)

    assert meta["court_detection_success"], f"{fixture_id}: detection failed ({meta.get('error')})"
    assert out_mask is not None and out_mask.shape == golden.shape
    score = iou(out_mask, golden)
    assert score >= 1.0 - DET_TOL, f"{fixture_id}: IoU={score:.5f} < {1.0 - DET_TOL}"
