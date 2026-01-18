from __future__ import annotations

import yaml

from training.io.config import load_config


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
