from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from training.io.config import load_config  # noqa: E402


def test_load_config_deep_merge(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_root": "data/custom",
                "wandb": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))
    assert cfg["data_root"] == "data/custom"
    assert cfg["wandb"]["enabled"] is True
    assert cfg["wandb"]["project"] is None
    assert cfg["wandb"]["entity"] is None
    assert cfg["videos"] == []
    assert cfg["config_path"] == str(cfg_path)


def test_comma_separated_overlays_merge_left_to_right(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"train": {"lr": 1e-3, "epochs": 10}, "videos": ["a"]}), encoding="utf-8")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"train": {"epochs": 40}, "videos": ["b"]}), encoding="utf-8")

    cfg = load_config(f"{base}, {overlay}")
    assert cfg["train"] == {"lr": 1e-3, "epochs": 40}    # nested keys merge
    assert cfg["videos"] == ["b"]                        # lists replace
