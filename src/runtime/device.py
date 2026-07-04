from __future__ import annotations

import logging
import os
from typing import Literal, Optional

DeviceName = Literal["cuda", "mps", "cpu"]

_DEVICE_ORDER: tuple[DeviceName, ...] = ("cuda", "mps", "cpu")


def _torch_available() -> bool:
    try:
        import torch  # noqa: WPS433 — optional at import time for tests
    except ImportError:
        return False
    return True


def detect_available_devices() -> list[DeviceName]:
    """Return acceleration backends available on this machine, in priority order."""
    if not _torch_available():
        return ["cpu"]
    import torch

    available: list[DeviceName] = []
    if torch.cuda.is_available():
        available.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        available.append("mps")
    available.append("cpu")
    return available


def resolve_auto_device() -> DeviceName:
    """Pick the best device: CUDA, then MPS, then CPU."""
    devices = detect_available_devices()
    for name in _DEVICE_ORDER:
        if name in devices:
            return name
    return "cpu"


def resolve_pose_device(explicit: Optional[str] = None, *, read_env: bool = True) -> DeviceName:
    """Resolve pose/YOLO device from explicit choice, env, or auto priority."""
    valid = set(_DEVICE_ORDER)
    if explicit:
        choice = explicit.strip().lower()
        if choice in {"", "auto"}:
            explicit = None
        elif choice in valid:
            return choice  # type: ignore[return-value]
        else:
            raise ValueError(f"Unsupported device '{explicit}'. Choose auto, cuda, mps, or cpu.")

    if read_env:
        env_val = os.environ.get("POSE_DEVICE", "").strip().lower()
        if env_val in valid:
            return env_val  # type: ignore[return-value]

    return resolve_auto_device()


def prefer_cpu_over_mps_for_pose(device: DeviceName, model_path: str, *, warn: bool = True) -> DeviceName:
    """Avoid auto-selected MPS for YOLO pose models with known backend issues."""
    if device == "mps" and "pose" in model_path.lower():
        if warn:
            logging.warning(
                "POSE: defaulting to cpu due to known MPS pose performance issues; "
                "set yolo_device=mps to force MPS."
            )
        return "cpu"
    return device


def apply_pose_device(
    explicit: Optional[str] = None,
    *,
    model_path: str,
    set_env: bool = True,
    read_env: Optional[bool] = None,
) -> DeviceName:
    """Set POSE_DEVICE for downstream PoseExtractor and return the resolved device."""
    user_explicit = bool(explicit and explicit.strip().lower() not in ("", "auto"))
    use_env = set_env if read_env is None else read_env
    device = resolve_pose_device(explicit, read_env=use_env)
    if not user_explicit:
        device = prefer_cpu_over_mps_for_pose(device, model_path)
    if set_env:
        os.environ["POSE_DEVICE"] = device
    return device
