"""Training loop for the TCN boundary-heatmap head (E2EHeatmapLoss).

Each epoch is scored with the stitched six-bin evaluation on a CPU-reloaded copy
of the model, using the shipped runtime decode. Checkpoint selection and early
stopping follow `selection_metric` ("val_loss" | "acceptable" | "good_weighted" |
"f1"). Writes a manifest.json into the run dir at the end."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from training.dataset.hdf5_dataset import Hdf5SequenceDataset
from training.eval.heatmap_evaluator import HeatmapDecodeConfig, evaluate_heatmap_model
from training.metrics.segments6 import SixBinConfig
from training.models.heatmap_tcn import TennisPointHeatmapTCN
from training.train.heatmap_loss import E2EHeatmapLoss, HeatmapLossConfig
from training.train.loop import _default_pos_weight, _resolve_device, _seed_everything, _to_jsonable

logger = logging.getLogger(__name__)

SELECTION_METRICS = ("val_loss", "acceptable", "good_weighted", "f1")


def build_heatmap_model(train_cfg: Dict[str, Any], input_size: int) -> TennisPointHeatmapTCN:
    backbone = str(train_cfg.get("heatmap_backbone", "tcn")).lower()
    if backbone != "tcn":
        raise ValueError(f"Unknown heatmap_backbone: {backbone!r} (only tcn is supported)")
    stem_hidden = train_cfg.get("heatmap_tcn_stem_hidden")
    return TennisPointHeatmapTCN(
        input_size=int(input_size),
        hidden_size=int(train_cfg.get("hidden_size", 64)),
        levels=int(train_cfg.get("heatmap_tcn_levels", 5)),
        kernel_size=int(train_cfg.get("heatmap_tcn_kernel_size", 3)),
        dropout=float(train_cfg.get("dropout", 0.2)),
        head=str(train_cfg.get("heatmap_head", "mlp")),
        stem_hidden=None if stem_hidden is None else int(stem_hidden),
    )


def load_heatmap_model(checkpoint_path: Path, train_cfg: Dict[str, Any]) -> TennisPointHeatmapTCN:
    """Rebuild a TCN from a checkpoint on CPU, in eval mode.

    Checkpoints carry their constructor kwargs under "arch". Older ones (the
    v0.6 CV runs) do not, so their shape is read off the state dict, with
    head/kernel/dropout from the run's train config.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    arch = ckpt.get("arch") or _arch_from_state(state, train_cfg)
    model = TennisPointHeatmapTCN(**arch)
    model.load_state_dict(state)
    return model.eval()


def _arch_from_state(state: Dict[str, torch.Tensor], train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "input_proj.0.weight" in state:  # two-layer stem: input -> stem_hidden -> hidden
        stem = state["input_proj.0.weight"]
        input_size, stem_hidden = int(stem.shape[1]), int(stem.shape[0])
        hidden_size = int(state["input_proj.2.weight"].shape[0])
    else:
        proj = state["input_proj.weight"]  # (hidden, input_size, 1)
        input_size, hidden_size, stem_hidden = int(proj.shape[1]), int(proj.shape[0]), None
    levels = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks.") and ".conv1." in k)
    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "levels": levels,
        "kernel_size": int(state["blocks.0.conv1.weight"].shape[-1]),
        "dropout": float(train_cfg.get("dropout", 0.2)),
        "head": "linear" if "fc.weight" in state else "mlp",
        "stem_hidden": stem_hidden,
    }


def _acceptable_f1(metrics: Dict[str, float]) -> float:
    """F1 over the (good + decent) set: harmonic mean of precision and recall.

    `acceptable` (the historical default) is (good+decent)/n_gt -- recall with no
    precision term, so a model that floods the timeline with predictions scores
    well on it. Selecting on F1 makes over-prediction cost something.
    """
    ok = float(metrics.get("n_good", 0.0)) + float(metrics.get("n_decent", 0.0))
    n_gt = float(metrics.get("n_gt", 0.0))
    n_pred = float(metrics.get("n_pred", 0.0))
    recall = ok / n_gt if n_gt else 0.0
    precision = ok / n_pred if n_pred else 0.0
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _selection_score(selection_metric: str, val_loss: float, val_metrics: Dict[str, float]) -> float:
    if selection_metric == "val_loss":
        return -val_loss
    if selection_metric == "good_weighted":
        return 2.0 * float(val_metrics.get("share_good", 0.0)) + float(val_metrics.get("share_decent", 0.0))
    if selection_metric == "f1":
        return _acceptable_f1(val_metrics)
    return float(val_metrics.get("share_good", 0.0)) + float(val_metrics.get("share_decent", 0.0))


def train_heatmap(dataset_dir: Path, run_dir: Path, config: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 1337))
    _seed_everything(seed)

    train_path = dataset_dir / "train.h5"
    val_path = dataset_dir / "val.h5"
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val dataset not found: {val_path}")

    train_ds = Hdf5SequenceDataset(train_path)
    device = _resolve_device(config.get("device"))
    model = build_heatmap_model(config, train_ds.feature_dim).to(device)

    if config.get("pos_weight") is None:
        pos_weight_value = _default_pos_weight(train_ds)
        logger.info("Derived pos_weight=%.4f from train positive rate", pos_weight_value)
    else:
        pos_weight_value = float(config["pos_weight"])

    fps = float(config.get("fps", 5.0))
    loss_cfg = HeatmapLossConfig.from_train_cfg(config, fps, pos_weight_value)
    criterion = E2EHeatmapLoss(loss_cfg).to(device)
    criterion_cpu = E2EHeatmapLoss(loss_cfg)  # for the CPU-side per-epoch eval
    decode_cfg = HeatmapDecodeConfig.from_train_cfg(config, fps)
    six_bin_cfg = SixBinConfig()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 0.01)
    )
    # Optional single-cycle cosine decay from the configured peak lr -> eta_min over
    # all epochs (no warm restarts). Stepped once per epoch.
    total_epochs = int(config.get("epochs", 30))
    lr_schedule = str(config.get("lr_schedule", "none")).lower()
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=float(config.get("lr_eta_min", 0.0))
        )
    elif lr_schedule != "none":
        raise ValueError(f"Unknown lr_schedule: {lr_schedule!r} (expected none | cosine)")

    batch_size = int(config.get("batch_size", 32))
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)

    early_stopping_patience = max(0, int(config.get("early_stopping_patience", 0)))
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))
    selection_metric = str(config.get("selection_metric", "acceptable"))
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")

    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []
    started_at = time.time()

    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not config_path.exists():
        # The resolved loss (incl. a derived pos_weight) is saved so pipeline eval
        # scores the test split with exactly the loss the run trained with.
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable({**config, "loss": asdict(loss_cfg)}), handle, indent=2)

    logger.info("Heatmap training setup: device=%s batch=%d loss=%s", device, batch_size, loss_cfg)

    for epoch in range(1, total_epochs + 1):
        epoch_started = time.time()
        model.train()
        running = {"loss": 0.0, "loss_cls": 0.0, "loss_start": 0.0, "loss_end": 0.0}
        batches = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            p_logits, s_logits, e_logits = model(features)
            loss, components = criterion(p_logits, s_logits, e_logits, targets)
            loss.backward()
            optimizer.step()
            running["loss"] += float(loss.item())
            for key, value in components.items():
                running[key] += value
            batches += 1

        train_means = {f"train_{k}": v / max(batches, 1) for k, v in running.items()}
        # Score a CPU-reloaded copy rather than the live (possibly MPS) model:
        # CPU eval of the saved weights is exactly what gets exported and shipped.
        eval_model = TennisPointHeatmapTCN(**model.arch())
        eval_model.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        val_metrics, val_loss = evaluate_heatmap_model(
            eval_model, val_path, torch.device("cpu"), criterion_cpu, decode_cfg, six_bin_cfg, batch_size=batch_size
        )
        del eval_model

        log_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **train_means,
            "val_loss": val_loss,
            **val_metrics,
            "epoch_seconds": round(time.time() - epoch_started, 1),
        }
        history.append(log_row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_row) + "\n")

        logger.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f bal_acc=%.4f | shares good=%.3f decent=%.3f bad_seg=%.3f poor=%.3f fp=%.3f fn=%.3f",
            epoch,
            train_means["train_loss"],
            val_loss,
            float(val_metrics.get("balanced_accuracy", 0.0)),
            val_metrics.get("share_good", 0.0),
            val_metrics.get("share_decent", 0.0),
            val_metrics.get("share_bad_segmentation", 0.0),
            val_metrics.get("share_poor_recognition", 0.0),
            val_metrics.get("share_false_positive", 0.0),
            val_metrics.get("share_false_negative", 0.0),
        )

        score = _selection_score(selection_metric, val_loss, val_metrics)
        if score > (best_score + early_stopping_min_delta):
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(run_dir / "checkpoints" / "best.pth", model, optimizer, epoch, log_row)
        else:
            epochs_without_improvement += 1
        _save_checkpoint(run_dir / "checkpoints" / "last.pth", model, optimizer, epoch, log_row)
        if config.get("save_every_n") and epoch % int(config["save_every_n"]) == 0:
            _save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch}.pth", model, optimizer, epoch, log_row)

        if scheduler is not None:
            scheduler.step()

        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d (no %s improvement for %d epochs)",
                epoch,
                selection_metric,
                epochs_without_improvement,
            )
            break

    write_run_manifest(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        config=config,
        model=model,
        loss_cfg=loss_cfg,
        decode_cfg=decode_cfg,
        history=history,
        best_epoch=best_epoch,
        selection_metric=selection_metric,
        total_seconds=time.time() - started_at,
    )


def evaluate_heatmap_run(
    run_dir: Path, test_path: Path, train_cfg: Dict[str, Any], fps: float
) -> Tuple[Dict[str, float], float]:
    """Score a finished run's best.pth on `test_path` with the run's own loss
    and decode settings.

    The loss comes from the run's config.json "loss" block when present, so a
    pos_weight derived at train time is reused rather than re-guessed.
    """
    config_path = run_dir / "config.json"
    saved = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if saved.get("loss"):
        # Filter to current fields: older runs also saved the removed time_* knobs.
        known = {f.name for f in fields(HeatmapLossConfig)}
        loss_cfg = HeatmapLossConfig(**{k: v for k, v in saved["loss"].items() if k in known})
    else:
        pos_weight = train_cfg.get("pos_weight")
        loss_cfg = HeatmapLossConfig.from_train_cfg(train_cfg, fps, 3.0 if pos_weight is None else pos_weight)
    model = load_heatmap_model(run_dir / "checkpoints" / "best.pth", train_cfg)
    return evaluate_heatmap_model(
        model,
        test_path,
        device=torch.device("cpu"),
        criterion=E2EHeatmapLoss(loss_cfg),
        decode_cfg=HeatmapDecodeConfig.from_train_cfg(train_cfg, fps),
        six_bin_cfg=SixBinConfig(),
    )


def write_run_manifest(
    *,
    run_dir: Path,
    dataset_dir: Path,
    config: Dict[str, Any],
    model: TennisPointHeatmapTCN,
    loss_cfg: HeatmapLossConfig,
    decode_cfg: HeatmapDecodeConfig,
    history: List[Dict[str, Any]],
    best_epoch: int,
    selection_metric: str,
    total_seconds: float,
) -> Path:
    dataset_manifest: Dict[str, Any] = {}
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            dataset_manifest = json.load(handle)

    selected = next((row for row in history if row.get("epoch") == best_epoch), None)
    best_acceptable = None
    if history:
        best_acceptable = max(
            history,
            key=lambda r: float(r.get("share_good", 0.0)) + float(r.get("share_decent", 0.0)),
        )

    manifest = {
        "manifest_version": 1,
        "artifact": {
            "created_at_iso": datetime.now(timezone.utc).isoformat(),
            "run_id": run_dir.name,
            "model_type": "e2e_heatmap",
        },
        "model": {
            "architecture": "TennisPointHeatmapTCN",
            "backbone": "tcn",
            **model.arch(),
            "receptive_field_frames": model.receptive_field,
            "outputs": ["pointness_logit", "start_heatmap_logit", "end_heatmap_logit"],
        },
        "loss": asdict(loss_cfg),
        "training": {
            **{k: v for k, v in config.items() if not isinstance(v, (dict, list))},
            "selection_metric": selection_metric,
            "epochs_run": len(history),
            "selected_epoch": best_epoch,
            "total_seconds": round(total_seconds, 1),
            "mean_epoch_seconds": round(
                sum(r.get("epoch_seconds", 0.0) for r in history) / max(len(history), 1), 1
            ),
        },
        # Every decode field, so a manifest reproduces the decode exactly;
        # runtime_params is the subset the shipped runtime reads.
        "decode": {
            "method": "gaussian_peak_softargmax",
            **asdict(decode_cfg),
            "runtime_params": decode_cfg.runtime_params(),
        },
        "data": {
            "dataset_dir": str(dataset_dir),
            "feature_set": dataset_manifest.get("feature_set"),
            "dataset_config": dataset_manifest.get("config"),
            "splits": dataset_manifest.get("splits"),
        },
        "metrics": {
            "selected_epoch_metrics": selected,
            "best_acceptable_epoch_metrics": best_acceptable,
            "last_epoch_metrics": history[-1] if history else None,
        },
        "source_run": {
            "checkpoint_path": str(run_dir / "checkpoints" / "best.pth"),
            "scaler_path": str(dataset_dir / "scaler.joblib"),
            **_git_info(),
        },
    }

    out_path = run_dir / "manifest.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(manifest), handle, indent=2)
    logger.info("Wrote run manifest to %s", out_path)
    return out_path


def _git_info() -> Dict[str, Any]:
    def _run(args: List[str]) -> str:
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


def _save_checkpoint(
    path: Path, model: TennisPointHeatmapTCN, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, Any]
) -> None:
    # Serialize CPU clones and verify the written file: torch.save of MPS-resident
    # tensors has been observed to write corrupted weights.
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "metrics": metrics,
            "arch": model.arch(),
        },
        str(path),
    )
    written = torch.load(str(path), map_location="cpu")["model_state_dict"]
    for key, value in model_state.items():
        if not torch.equal(written[key], value):
            raise RuntimeError(f"Checkpoint verification failed for {path} (tensor {key})")
