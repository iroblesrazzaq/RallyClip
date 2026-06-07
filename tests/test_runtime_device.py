from __future__ import annotations

import types

import pytest


def test_resolve_auto_device_prefers_cuda(monkeypatch):
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    from runtime.device import resolve_auto_device

    assert resolve_auto_device() == "cuda"


def test_resolve_auto_device_falls_back_to_mps(monkeypatch):
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    from runtime.device import resolve_auto_device

    assert resolve_auto_device() == "mps"


def test_resolve_auto_device_falls_back_to_cpu(monkeypatch):
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
    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    monkeypatch.delenv("POSE_DEVICE", raising=False)
    from runtime.device import apply_pose_device

    device = apply_pose_device(None)
    assert device == "cpu"
    import os

    assert os.environ["POSE_DEVICE"] == "cpu"
