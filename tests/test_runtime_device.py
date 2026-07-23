from __future__ import annotations

import types


def _disable_coreml(monkeypatch):
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "coreml_pose_available", lambda: False)


def _disable_cuda_ort(monkeypatch):
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: False)


def test_resolve_auto_device_prefers_cuda(monkeypatch):
    _disable_cuda_ort(monkeypatch)
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    from runtime.device import resolve_auto_device

    assert resolve_auto_device() == "cuda"


def test_resolve_auto_device_falls_back_to_mps(monkeypatch):
    _disable_coreml(monkeypatch)
    _disable_cuda_ort(monkeypatch)
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    from runtime.device import resolve_auto_device

    assert resolve_auto_device() == "mps"


def test_resolve_auto_device_falls_back_to_cpu(monkeypatch):
    _disable_coreml(monkeypatch)
    _disable_cuda_ort(monkeypatch)
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    from runtime.device import resolve_auto_device

    assert resolve_auto_device() == "cpu"


def test_resolve_pose_device_honors_explicit_choice(monkeypatch):
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    from runtime.device import resolve_pose_device

    assert resolve_pose_device("cpu") == "cpu"


def test_apply_pose_device_sets_env(monkeypatch):
    _disable_coreml(monkeypatch)
    _disable_cuda_ort(monkeypatch)
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    monkeypatch.delenv("POSE_DEVICE", raising=False)
    from runtime.device import apply_pose_device

    device = apply_pose_device(None, model_path="yolov8n-pose.pt")
    assert device == "cpu"
    import os

    assert os.environ["POSE_DEVICE"] == "cpu"


def test_apply_pose_device_honors_explicit_mps(monkeypatch):
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    monkeypatch.delenv("POSE_DEVICE", raising=False)
    from runtime.device import apply_pose_device

    device = apply_pose_device("mps", model_path="yolov8n-pose.pt")
    assert device == "mps"


def test_apply_pose_device_gui_mode_ignores_ambient_env(monkeypatch):
    """GUI jobs use set_env=False and must not inherit shell POSE_DEVICE."""
    import os

    _disable_coreml(monkeypatch)
    _disable_cuda_ort(monkeypatch)

    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    monkeypatch.setenv("POSE_DEVICE", "cuda")

    from runtime.device import apply_pose_device

    device = apply_pose_device(None, model_path="yolov8n-pose.pt", set_env=False)
    assert device == "cpu"
    assert os.environ["POSE_DEVICE"] == "cuda"


def test_resolve_pose_device_accepts_coreml():
    from runtime.device import resolve_pose_device

    assert resolve_pose_device("coreml") == "coreml"


def test_auto_device_prefers_coreml_when_available(monkeypatch):
    """The shipped Apple-silicon app should get CoreML without touching
    settings; CPU is the fallback, not the auto pick (no torch here)."""
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "_torch_available", lambda: False)
    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: False)
    monkeypatch.setattr(device_mod, "coreml_pose_available", lambda: True)
    assert "coreml" in device_mod.detect_available_devices()
    assert device_mod.resolve_auto_device() == "coreml"


def test_auto_device_without_coreml_stays_cpu(monkeypatch):
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "_torch_available", lambda: False)
    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: False)
    monkeypatch.setattr(device_mod, "coreml_pose_available", lambda: False)
    assert device_mod.resolve_auto_device() == "cpu"


def test_coreml_unavailable_off_apple_silicon(monkeypatch):
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod.sys, "platform", "linux")
    assert device_mod.coreml_pose_available() is False


def test_detect_available_devices_includes_ort_cuda_without_torch(monkeypatch):
    """Stock torch-free runtime must still surface CUDA when ORT has the EP."""
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "_torch_available", lambda: False)
    monkeypatch.setattr(device_mod, "coreml_pose_available", lambda: False)
    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: True)
    assert device_mod.detect_available_devices() == ["cuda", "cpu"]
    assert device_mod.resolve_auto_device() == "cuda"


def test_detect_available_devices_omits_cuda_without_ort_or_torch(monkeypatch):
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "_torch_available", lambda: False)
    monkeypatch.setattr(device_mod, "coreml_pose_available", lambda: False)
    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: False)
    assert "cuda" not in device_mod.detect_available_devices()


def test_cuda_pose_available_reads_ort_providers(monkeypatch):
    import runtime.device as device_mod

    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    assert device_mod.cuda_pose_available() is True

    fake_ort_cpu = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"]
    )
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort_cpu)
    assert device_mod.cuda_pose_available() is False


def test_apply_pose_device_auto_pt_demotes_ort_only_cuda(monkeypatch):
    """Auto + .pt must not keep CUDA when only ORT exposes the EP (no torch CUDA)."""
    import os

    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "coreml_pose_available", lambda: False)
    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: True)
    monkeypatch.setattr(device_mod, "torch_cuda_available", lambda: False)
    monkeypatch.setattr(device_mod, "_torch_available", lambda: False)
    monkeypatch.delenv("POSE_DEVICE", raising=False)

    device = device_mod.apply_pose_device(None, model_path="yolov8n-pose.pt")
    assert device == "cpu"
    assert os.environ["POSE_DEVICE"] == "cpu"


def test_apply_pose_device_explicit_cuda_kept_for_pt_resolver(monkeypatch):
    """Explicit cuda is still returned from apply; PoseExtractor degrades at load."""
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "torch_cuda_available", lambda: False)
    monkeypatch.delenv("POSE_DEVICE", raising=False)
    assert device_mod.apply_pose_device("cuda", model_path="yolov8n-pose.pt") == "cuda"


def test_apply_pose_device_auto_onnx_keeps_ort_cuda(monkeypatch):
    import runtime.device as device_mod

    monkeypatch.setattr(device_mod, "coreml_pose_available", lambda: False)
    monkeypatch.setattr(device_mod, "cuda_pose_available", lambda: True)
    monkeypatch.setattr(device_mod, "torch_cuda_available", lambda: False)
    monkeypatch.setattr(device_mod, "_torch_available", lambda: False)
    monkeypatch.delenv("POSE_DEVICE", raising=False)
    assert device_mod.apply_pose_device(None, model_path="yolov8n-pose.onnx") == "cuda"
