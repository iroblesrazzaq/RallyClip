"""Feature set v2 lives in features.v2 (shared with the runtime); re-exported here."""
from features.v2 import (  # noqa: F401
    DEFAULT_SLOTS,
    NEARFAR_SLOTS,
    SLOT_GROUPS,
    FeatureSetV2,
    FeatureSetV2NearFar,
    V2FeatureStream,
    kept_columns,
)
