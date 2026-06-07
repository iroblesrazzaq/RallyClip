"""Shared runtime utilities for CLI, GUI, and desktop packaging."""

from .device import detect_available_devices, resolve_pose_device
from .paths import resolve_frontend_dir

__all__ = [
    "detect_available_devices",
    "resolve_pose_device",
    "resolve_frontend_dir",
]
