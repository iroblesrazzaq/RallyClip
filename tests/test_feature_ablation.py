import numpy as np
import pytest

from training.features.v2 import NEARFAR_SLOTS, SLOT_GROUPS, FeatureSetV2, kept_columns


def test_groups_cover_the_slot_block():
    assert sum(width for _, width in SLOT_GROUPS) == FeatureSetV2.per_slot_dim()


def test_no_drop_keeps_everything():
    np.testing.assert_array_equal(kept_columns(NEARFAR_SLOTS), np.arange(290))


def test_drop_derivatives_and_far_slot():
    derivs = ("velocity", "acceleration", "keypoint_vel", "keypoint_accel")
    assert len(kept_columns(NEARFAR_SLOTS, derivs)) == 2 * 73
    keep = kept_columns(NEARFAR_SLOTS, drop_slots=("far",))
    np.testing.assert_array_equal(keep, np.arange(145))


def test_drop_box_conf_is_last_column_of_each_slot():
    keep = kept_columns(NEARFAR_SLOTS, ("box_conf",))
    assert 144 not in keep and 289 not in keep and len(keep) == 288


def test_unknown_group_raises():
    with pytest.raises(ValueError):
        kept_columns(NEARFAR_SLOTS, ("speed",))
