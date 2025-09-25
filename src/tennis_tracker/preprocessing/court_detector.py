# Temporary compatibility wrapper to expose court detector via the core package
# without duplicating the large implementation file.

from data_scripts.court_detector import CourtDetector, filter_players_by_playable_area  # type: ignore

__all__ = [
    "CourtDetector",
    "filter_players_by_playable_area",
]


