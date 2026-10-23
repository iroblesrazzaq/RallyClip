from __future__ import annotations

import pytest

from training.courts.anchors import court_anchor_times


@pytest.mark.parametrize("duration", [0.8, 12.0, 15.0, 19.9, 600.0])
def test_anchors_stay_inside_the_clip_middle_first(duration):
    times = court_anchor_times(duration)
    assert times[0] == pytest.approx(duration / 2, abs=0.01)
    assert all(0 <= t < duration for t in times)


@pytest.mark.parametrize("duration", [None, 0.0])
def test_unknown_duration_falls_back_to_fixed_times(duration):
    assert court_anchor_times(duration) == [60.0, 45.0, 90.0, 30.0, 120.0]
