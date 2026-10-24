"""TCN heatmap backbone: output contract, receptive field, factory, checkpoint reload."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.models.heatmap_tcn import TennisPointHeatmapTCN  # noqa: E402
from training.train.heatmap_loop import (  # noqa: E402
    _save_checkpoint,
    build_heatmap_model,
    load_heatmap_model,
)

INPUT_SIZE, SEQ_LEN = 362, 100


def _tcn(**kw):
    return TennisPointHeatmapTCN(input_size=INPUT_SIZE, hidden_size=64, **kw)


def test_forward_shapes_match_the_three_logit_contract():
    model = _tcn()
    p, s, e = model(torch.randn(4, SEQ_LEN, INPUT_SIZE))
    for out in (p, s, e):
        assert out.shape == (4, SEQ_LEN)
        assert torch.isfinite(out).all()


def test_receptive_field_covers_the_training_window():
    # RF = 1 + 2*(k-1)*(2**levels - 1); defaults must span the 100-frame (20s) window.
    model = _tcn(levels=5, kernel_size=3)
    assert model.receptive_field == 1 + 2 * 2 * (2 ** 5 - 1) == 125
    assert model.receptive_field >= SEQ_LEN
    assert _tcn(levels=4, kernel_size=3).receptive_field == 61


def test_is_non_causal_so_it_sees_the_future_like_the_bilstm():
    """The backbone it replaces is bidirectional; a causal TCN would be a
    handicap. Perturbing frame t must move outputs on BOTH sides of t."""
    torch.manual_seed(0)
    model = _tcn().eval()
    x = torch.randn(1, SEQ_LEN, INPUT_SIZE)
    with torch.no_grad():
        base = model(x)[0]
        bumped = x.clone()
        bumped[:, 60, :] += 5.0
        after = model(bumped)[0]
    delta = (base - after).abs()
    assert delta[:, 50].max() > 0, "must influence earlier frames (non-causal)"
    assert delta[:, 70].max() > 0, "must influence later frames"


def test_factory_builds_from_train_config_and_rejects_other_backbones():
    cfg = {"hidden_size": 32, "heatmap_tcn_levels": 4, "heatmap_tcn_kernel_size": 5, "dropout": 0.1}
    model = build_heatmap_model(cfg, INPUT_SIZE)
    assert model.arch() == {
        "input_size": INPUT_SIZE, "hidden_size": 32, "levels": 4, "kernel_size": 5,
        "dropout": 0.1, "head": "mlp", "stem_hidden": None,
    }
    with pytest.raises(ValueError, match="tcn"):
        build_heatmap_model({"heatmap_backbone": "lstm"}, INPUT_SIZE)


@pytest.mark.parametrize("stem_hidden,head", [(None, "mlp"), (128, "mlp"), (None, "linear")])
def test_checkpoint_reloads_with_and_without_stored_arch(tmp_path, stem_hidden, head):
    """New checkpoints carry "arch"; the v0.6 CV checkpoints do not, so their
    shape is read off the state dict. Both must rebuild the same network."""
    cfg = {"hidden_size": 32, "heatmap_tcn_levels": 3, "heatmap_tcn_kernel_size": 3,
           "heatmap_tcn_stem_hidden": stem_hidden, "heatmap_head": head, "dropout": 0.2}
    model = build_heatmap_model(cfg, INPUT_SIZE).eval()
    path = tmp_path / "best.pth"
    _save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), epoch=1, metrics={})
    legacy = tmp_path / "legacy.pth"
    torch.save({"model_state_dict": torch.load(path)["model_state_dict"]}, legacy)

    x = torch.randn(1, 40, INPUT_SIZE)
    with torch.no_grad():
        want = model(x)[0]
    for ckpt in (path, legacy):
        reloaded = load_heatmap_model(ckpt, cfg)
        assert reloaded.arch() == model.arch()
        with torch.no_grad():
            assert torch.allclose(reloaded(x)[0], want, atol=1e-6)


def test_eval_decodes_with_the_settings_the_run_trained_with(tmp_path, monkeypatch):
    import json

    from training.train import heatmap_loop

    (tmp_path / "config.json").write_text(json.dumps({"heatmap_peak_threshold": 0.3}))
    seen = {}
    monkeypatch.setattr(heatmap_loop, "load_heatmap_model", lambda path, cfg: seen.setdefault("model_cfg", cfg))
    monkeypatch.setattr(heatmap_loop, "evaluate_heatmap_model",
                        lambda model, path, **kw: seen.setdefault("decode", kw["decode_cfg"]))

    # The YAML was edited after training; eval must still score the run's decoder.
    heatmap_loop.evaluate_heatmap_run(tmp_path, tmp_path / "test.h5", {"heatmap_peak_threshold": 0.1}, 5.0)
    assert seen["decode"].peak_threshold == 0.3
    assert seen["model_cfg"]["heatmap_peak_threshold"] == 0.3


@pytest.mark.parametrize("present, scored", [(["best.pth", "last.pth"], "best.pth"), (["last.pth"], "last.pth")])
def test_eval_scores_last_when_nothing_was_selected(tmp_path, monkeypatch, present, scored):
    from training.train import heatmap_loop

    (tmp_path / "checkpoints").mkdir()
    for name in present:
        (tmp_path / "checkpoints" / name).write_bytes(b"")
    seen = []
    monkeypatch.setattr(heatmap_loop, "load_heatmap_model", lambda path, cfg: seen.append(path.name))
    monkeypatch.setattr(heatmap_loop, "evaluate_heatmap_model", lambda model, path, **kw: None)
    heatmap_loop.evaluate_heatmap_run(tmp_path, tmp_path / "test.h5", {}, 5.0)
    assert seen == [scored]


def test_eval_refuses_another_fps(tmp_path):
    import json

    from training.train import heatmap_loop

    (tmp_path / "config.json").write_text(json.dumps({"fps": 5.0}))
    with pytest.raises(ValueError, match="trained at 5.0 fps"):
        heatmap_loop.evaluate_heatmap_run(tmp_path, tmp_path / "test.h5", {}, 10.0)
