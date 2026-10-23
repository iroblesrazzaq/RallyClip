from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from training.courts import cache as court_cache  # noqa: E402
from training.courts.cache import CourtCacheResult, CourtMaskCache  # noqa: E402


def _result(policy):
    detector = {"model_path": "m", "target_time": 1.0}
    if policy is not None:
        detector["anchor_policy"] = policy
    return CourtCacheResult(success=True, mask=np.ones((2, 2), np.uint8), metadata={"detector": detector}, lines=None)


@pytest.mark.parametrize("cached_policy, want_target, recompute", [
    ("middle", None, False),
    (60, 60, False),
    ("middle", 60, True),
    (60, None, True),
    (None, 60, False),  # legacy cache without the field: reused with a warning
])
def test_cache_recomputes_only_on_anchor_policy_mismatch(tmp_path, monkeypatch, cached_policy, want_target, recompute):
    monkeypatch.setattr(court_cache, "_video_metadata", lambda _p: {})
    cache = CourtMaskCache(model_path="m", target_time=want_target)
    video = tmp_path / "clip.mp4"
    cache.save(cache.cache_path(tmp_path, video), _result(cached_policy))

    calls = []

    def fake_compute(_video):
        calls.append(1)
        return _result(cache.anchor_policy)

    monkeypatch.setattr(cache, "compute", fake_compute)
    cache.get_or_create(tmp_path, video)
    assert bool(calls) is recompute
