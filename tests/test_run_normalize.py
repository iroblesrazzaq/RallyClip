"""_run_normalize: where it finds inputs, and when it refuses to replace a video."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from training import pipeline  # noqa: E402
from training.paths import raw_videos_dir  # noqa: E402


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, "normalize_video", lambda src, dst, cfg: seen.append((src, dst)))
    return seen


def _config(root, **normalize):
    return {"data_root": str(root), "normalize": {"mode": "all", **normalize}}


def test_legacy_raw_videos_dir_is_a_source(tmp_path, calls):
    (tmp_path / "raw_videos").mkdir()
    (tmp_path / "raw_videos" / "m.mp4").write_bytes(b"")
    pipeline._run_normalize(_config(tmp_path))
    assert calls == [(tmp_path / "raw_videos" / "m.mp4", raw_videos_dir(tmp_path) / "m.mp4")]


def test_mismatched_normalized_video_is_not_silently_replaced(tmp_path, calls, monkeypatch):
    (tmp_path / "source_videos").mkdir()
    (tmp_path / "source_videos" / "m.mp4").write_bytes(b"")
    out = raw_videos_dir(tmp_path)
    out.mkdir(parents=True)
    (out / "m.mp4").write_bytes(b"")
    monkeypatch.setattr(pipeline, "is_canonical", lambda path, cfg: False)

    with pytest.raises(ValueError, match="does not match normalize settings"):
        pipeline._run_normalize(_config(tmp_path, fps=10))
    # normalize.overwrite alone would keep poses extracted from the old video.
    with pytest.raises(ValueError, match="does not match normalize settings"):
        pipeline._run_normalize(_config(tmp_path, fps=10, overwrite=True))
    assert calls == []

    pipeline._run_normalize({**_config(tmp_path, fps=10), "overwrite_all": True})
    assert len(calls) == 1


def test_mismatched_video_is_not_replaced_in_place(tmp_path, calls, monkeypatch):
    # No source dir: the normalized tree itself is the input.
    out = raw_videos_dir(tmp_path)
    out.mkdir(parents=True)
    (out / "m.mp4").write_bytes(b"")
    monkeypatch.setattr(pipeline, "is_canonical", lambda path, cfg: False)

    with pytest.raises(ValueError, match="does not match normalize settings"):
        pipeline._run_normalize(_config(tmp_path, fps=10))
    assert calls == []


def test_train_saves_the_feature_drops_for_the_sweep(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, "train_loop", lambda dataset_dir, run_dir, cfg: seen.update(cfg))
    pipeline._run_train({
        "data_root": str(tmp_path),
        "dataset": {"drop_feature_groups": ["derivatives"], "drop_slots": ["far"]},
    })
    assert seen["dataset"]["drop_feature_groups"] == ["derivatives"]
    assert seen["dataset"]["drop_slots"] == ["far"]


def test_absolute_source_paths_point_later_steps_at_the_normalized_copy(tmp_path, calls):
    source = tmp_path / "elsewhere" / "m.mp4"
    source.parent.mkdir()
    source.write_bytes(b"")
    config = {**_config(tmp_path), "videos": [str(source)]}
    pipeline._run_normalize(config)
    assert calls == [(source, raw_videos_dir(tmp_path) / "m.mp4")]
    assert config["videos"] == ["m.mp4"]
