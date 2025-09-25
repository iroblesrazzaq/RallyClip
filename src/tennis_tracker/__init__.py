"""
Tennis Tracker core library.

Subpackages:
- extraction: Pose extraction utilities
- preprocessing: Court filtering and player assignment
- features: Feature engineering from preprocessed data
- infer: Model loading and inference helpers
- segmentation: Video segmentation utilities
- pipeline: End-to-end orchestration helpers
"""

__all__ = [
    "extraction",
    "preprocessing",
    "features",
    "infer",
    "segmentation",
    "pipeline",
]

__version__ = "0.1.0"


