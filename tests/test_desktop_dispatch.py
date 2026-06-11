from __future__ import annotations

import sys
import types

import gui.desktop as desktop


def _stub_cli_main(monkeypatch, result: int = 0) -> dict:
    """Install a fake cli.main so the dispatch never imports torch/ultralytics."""
    calls: dict = {}

    def fake_main() -> int:
        calls["argv"] = list(sys.argv)
        return result

    cli_pkg = types.ModuleType("cli")
    cli_main_mod = types.ModuleType("cli.main")
    cli_main_mod.main = fake_main
    cli_pkg.main = cli_main_mod
    monkeypatch.setitem(sys.modules, "cli", cli_pkg)
    monkeypatch.setitem(sys.modules, "cli.main", cli_main_mod)
    return calls


def test_cli_flag_dispatches_with_flag_stripped(monkeypatch):
    calls = _stub_cli_main(monkeypatch, result=7)
    monkeypatch.setattr(sys, "argv", ["RallyClip", "--cli", "--video", "match.mp4", "--write-csv"])

    assert desktop.main() == 7
    assert calls["argv"] == ["RallyClip", "--video", "match.mp4", "--write-csv"]


def test_cli_flag_only_recognized_as_argv1(monkeypatch):
    calls = _stub_cli_main(monkeypatch)
    # Block the Qt path so the test can't start a real QApplication.
    monkeypatch.setitem(sys.modules, "PySide6", types.ModuleType("PySide6"))
    for mod in ("PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtWebEngineWidgets"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "argv", ["RallyClip", "--video", "x.mp4", "--cli"])

    assert desktop.main() == 1  # falls through to GUI path, which fails on stub Qt
    assert "argv" not in calls


def test_no_args_takes_gui_path(monkeypatch):
    calls = _stub_cli_main(monkeypatch)
    monkeypatch.setitem(sys.modules, "PySide6", types.ModuleType("PySide6"))
    for mod in ("PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtWebEngineWidgets"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "argv", ["RallyClip"])

    assert desktop.main() == 1
    assert "argv" not in calls
