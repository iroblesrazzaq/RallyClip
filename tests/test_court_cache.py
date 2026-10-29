from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from training.courts import cache as court_cache  # noqa: E402
from training.courts.cache import DETECTOR_KIND, CourtCacheResult, CourtMaskCache  # noqa: E402

POINTS = {"near_left": (240.0, 650.0), "near_right": (1040.0, 652.0),
          "far_left": (470.0, 200.0), "far_right": (810.0, 199.0)}


class _FakeCourt:
    frame_size = (1280, 720)

    def out_mask(self):
        mask = np.zeros((720, 1280), np.uint8)
        mask[:100] = 255
        return mask

    def geometry(self, out_size=None):
        return {"points_px": POINTS, "lines_ok": {"near_baseline": True}}


def _result(detector):
    return CourtCacheResult(success=True, mask=np.ones((2, 2), np.uint8), metadata={"detector": detector}, lines=None)


@pytest.mark.parametrize("detector, recompute", [
    ({"kind": DETECTOR_KIND, "model_path": "m"}, False),
    ({"model_path": "m", "target_time": 1.0, "anchor_policy": "middle"}, True),  # classic detector cache
    ({"kind": "something_else"}, True),
])
def test_cache_recomputes_a_cache_from_another_detector(tmp_path, monkeypatch, detector, recompute):
    monkeypatch.setattr(court_cache, "_video_metadata", lambda _p: {})
    cache = CourtMaskCache(model_path="m")
    video = tmp_path / "clip.mp4"
    cache.save(cache.cache_path(tmp_path, video), _result(detector))

    calls = []
    monkeypatch.setattr(cache, "compute", lambda _video: calls.append(1) or _result({"kind": DETECTOR_KIND}))
    cache.get_or_create(tmp_path, video)
    assert bool(calls) is recompute


def test_compute_stores_the_court_model_mask_and_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(court_cache, "_video_metadata", lambda _p: {"width": 1280, "height": 720})
    monkeypatch.setattr(court_cache, "detect_court", lambda video, model: _FakeCourt())
    cache = CourtMaskCache(model_path="court_lines.onnx")
    video = tmp_path / "clip.mp4"

    result = cache.get_or_create(tmp_path, video)
    assert result.success and result.mask[0, 0] == 255 and result.mask[200, 0] == 0
    meta = result.metadata
    assert meta["detector"] == {"kind": DETECTOR_KIND, "model_path": "court_lines.onnx"}
    assert meta["metadata"]["left_doubles_sideline"] == [[240.0, 650.0, 470.0, 200.0]]
    assert len(meta["metadata"]["extended_sidelines"]) == 2
    assert result.lines.shape == (5, 4)  # baseline, 2 sidelines, 2 padded sidelines

    reloaded = cache.load(cache.cache_path(tmp_path, video))
    assert reloaded.success and np.array_equal(reloaded.mask, result.mask)


def test_compute_without_a_court_is_a_failed_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(court_cache, "_video_metadata", lambda _p: {})
    monkeypatch.setattr(court_cache, "detect_court", lambda video, model: None)
    result = CourtMaskCache(model_path="m").compute(tmp_path / "clip.mp4")
    assert not result.success and result.mask is None
    assert result.metadata["metadata"]["court_detection_success"] is False


def test_compute_with_a_court_that_raises_is_a_failed_cache(tmp_path, monkeypatch):
    class _BadCourt(_FakeCourt):
        def geometry(self, out_size=None):
            raise ValueError("degenerate corners")

    monkeypatch.setattr(court_cache, "_video_metadata", lambda _p: {})
    monkeypatch.setattr(court_cache, "detect_court", lambda video, model: _BadCourt())
    result = CourtMaskCache(model_path="m").compute(tmp_path / "clip.mp4")
    assert not result.success and result.mask is None
    assert "degenerate corners" in result.metadata["metadata"]["error"]
