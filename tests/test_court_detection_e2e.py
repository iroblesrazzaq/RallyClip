"""Court-detection regression -- Flight 2: END-TO-END (live homography).

Runs the full pipeline on the source videos: ``extract_clean_frame`` (video seek
+ YOLO + RANSAC homography) -> detection -> mask, at the same timestamp the golden
was frozen from. The homography is probabilistic, so the tolerance is one order
of magnitude looser than Flight 1 (IoU >= 1 - 1e-2) -- still well within numerical
error, since the homography only repaints player-occluded regions away from the
court lines.

Skips when the source videos / YOLO weights / ultralytics are unavailable
(override locations via RALLYCLIP_COURT_VIDEO_DIR / RALLYCLIP_YOLO_WEIGHTS).
"""
from __future__ import annotations

import pytest

from helpers.court_fixtures import (
    FIX_DIR,
    iou,
    load_manifest,
    resolve_video_dir,
    resolve_yolo_weights,
)

cv2 = pytest.importorskip("cv2")
pytest.importorskip("ultralytics")

E2E_TOL = 1e-2  # one OOM looser than Flight 1's 1e-3; still ~numerical error
VIDEO_DIR = resolve_video_dir()
WEIGHTS = resolve_yolo_weights()
MANIFEST = load_manifest()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(VIDEO_DIR is None, reason="set RALLYCLIP_COURT_VIDEO_DIR to the court source videos"),
    pytest.mark.skipif(WEIGHTS is None, reason="set RALLYCLIP_YOLO_WEIGHTS to the YOLO pose weights"),
]


@pytest.fixture(scope="module")
def detector():
    import preprocessing.court_detector_impl as cdi
    return cdi.CourtDetector(yolo_model_path=str(WEIGHTS), conf=0.25)


@pytest.mark.parametrize("entry", MANIFEST, ids=[m["id"] for m in MANIFEST])
def test_detection_end_to_end(detector, entry):
    video = VIDEO_DIR / entry["video"]
    if not video.is_file():
        pytest.skip(f"source video missing: {video.name}")
    golden = cv2.imread(str(FIX_DIR / "masks" / f"{entry['id']}.png"), cv2.IMREAD_GRAYSCALE)
    assert golden is not None

    out_mask, _, meta = detector.process_video(str(video), target_time=entry["timestamp_s"])

    assert meta["court_detection_success"], f"{entry['id']}: e2e detection failed ({meta.get('error')})"
    assert out_mask is not None and out_mask.shape == golden.shape
    score = iou(out_mask, golden)
    assert score >= 1.0 - E2E_TOL, f"{entry['id']}: e2e IoU={score:.5f} < {1.0 - E2E_TOL}"
