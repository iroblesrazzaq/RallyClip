"""Court-detection anchor times for the training court cache."""
from __future__ import annotations

from typing import List, Optional

# Court-detection anchors as fractions of the clip, middle-first: mid-match is
# the most reliable place to find in-play footage with a settled camera.
COURT_ANCHOR_FRACTIONS = (0.5, 0.4, 0.6, 0.3, 0.7)


def court_anchor_times(duration_s: Optional[float]) -> List[float]:
    """Anchor times (seconds) to try for court detection, middle-first.

    Fractions of the duration keep every anchor inside a clip of any length; an
    unknown duration falls back to fixed times (the reader clamps past-the-end).
    """
    if duration_s and duration_s > 0:
        return [round(duration_s * f, 2) for f in COURT_ANCHOR_FRACTIONS]
    return [60.0, 45.0, 90.0, 30.0, 120.0]
