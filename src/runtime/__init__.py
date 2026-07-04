"""Shared runtime utilities for CLI, GUI, and desktop packaging."""

from .device import (
    apply_pose_device,
    detect_available_devices,
    resolve_auto_device,
    resolve_pose_device,
)
from .paths import resolve_frontend_dir

__all__ = [
    "apply_pose_device",
    "detect_available_devices",
    "resolve_auto_device",
    "resolve_pose_device",
    "resolve_frontend_dir",
]
