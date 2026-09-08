from __future__ import annotations

import sys
import types

import gui.desktop as desktop


def _stub_cli_main(monkeypatch, result: int = 0) -> dict:
    """Install a fake cli.main so the dispatch never imports torch/ultralytics."""
    calls: dict = {}

    def fake_main(*, force_cli: bool = False) -> int:
        calls["argv"] = list(sys.argv)
        calls["force_cli"] = force_cli
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
    assert calls["force_cli"] is True


def test_cli_flag_without_args_still_dispatches_to_cli(monkeypatch):
    calls = _stub_cli_main(monkeypatch, result=2)
    monkeypatch.setattr(sys, "argv", ["RallyClip", "--cli"])

    assert desktop.main() == 2
    assert calls["argv"] == ["RallyClip"]
    assert calls["force_cli"] is True


def _block_webview(monkeypatch):
    """Make `import webview` fail so the GUI path exits instead of opening a
    real window (dev venvs used for DMG builds have pywebview installed)."""
    monkeypatch.setitem(sys.modules, "webview", None)


def test_cli_flag_only_recognized_as_argv1(monkeypatch):
    calls = _stub_cli_main(monkeypatch)
    _block_webview(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["RallyClip", "--video", "x.mp4", "--cli"])

    assert desktop.main() == 1  # falls through to GUI path, which fails without pywebview
    assert "argv" not in calls


def test_no_args_takes_gui_path(monkeypatch):
    calls = _stub_cli_main(monkeypatch)
    _block_webview(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["RallyClip"])

    assert desktop.main() == 1
    assert "argv" not in calls


def test_backend_only_skips_webview(monkeypatch):
    launched: dict = {}

    def fake_launch(*, open_browser: bool = True) -> int:
        launched["open_browser"] = open_browser
        return 0

    import gui.app as gui_app

    monkeypatch.setattr(gui_app, "launch", fake_launch)
    _block_webview(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["RallyClip", "--backend-only"])

    assert desktop.main() == 0
    assert launched["open_browser"] is False
