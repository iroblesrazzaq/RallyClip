from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from training.dataset.hdf5_dataset import Hdf5SequenceDataset
from training.eval.evaluator import SegmentEvalConfig, evaluate_model
from training.models.lstm import TennisPointLSTM

logger = logging.getLogger(__name__)


def train(
    dataset_dir: Path,
    run_dir: Path,
    config: Dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.h5"
    val_path = dataset_dir / "val.h5"
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val dataset not found: {val_path}")

    train_ds = Hdf5SequenceDataset(train_path)
    val_ds = Hdf5SequenceDataset(val_path)

    device = _resolve_device(config.get("device"))
    model = TennisPointLSTM(
        input_size=train_ds.feature_dim,
        hidden_size=int(config.get("hidden_size", 128)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
        bidirectional=bool(config.get("bidirectional", True)),
        return_logits=True,
    ).to(device)

    pos_weight = torch.tensor([float(config.get("pos_weight", 3.0))], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 0.01))

    train_batch_size = int(config.get("train_batch_size", config.get("batch_size", 32)))
    eval_batch_size = int(config.get("eval_batch_size", train_batch_size))
    grad_accum_steps = max(1, int(config.get("grad_accum_steps", 1)))
    num_workers = int(config.get("num_workers", 0))
    effective_batch_size = train_batch_size * grad_accum_steps

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    threshold = float(config.get("threshold", 0.5))
    seg_cfg = config.get("segment_eval")
    segment_cfg = seg_cfg if isinstance(seg_cfg, SegmentEvalConfig) else SegmentEvalConfig()
    early_stopping_patience = max(0, int(config.get("early_stopping_patience", 0)))
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))

    best_metric = float("-inf")
    epochs_without_improvement = 0

    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not config_path.exists():
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(config), handle, indent=2)

    logger.info(
        "Training setup: device=%s train_batch=%d grad_accum=%d effective_batch=%d eval_batch=%d workers=%d early_stop_patience=%d min_delta=%.4f",
        device,
        train_batch_size,
        grad_accum_steps,
        effective_batch_size,
        eval_batch_size,
        num_workers,
        early_stopping_patience,
        early_stopping_min_delta,
    )

    wandb_run = _maybe_init_wandb(config, run_dir)
    try:
        for epoch in range(1, int(config.get("epochs", 10)) + 1):
            model.train()
            running_loss = 0.0
            batches = 0
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, (features, targets) in enumerate(train_loader, start=1):
                try:
                    features = features.to(device)
                    targets = targets.to(device)
                    logits = model(features)
                    loss = criterion(logits, targets)
                    running_loss += float(loss.item())
                    (loss / grad_accum_steps).backward()
                except RuntimeError as exc:
                    if _is_oom_error(exc):
                        optimizer.zero_grad(set_to_none=True)
                        _clear_device_cache(device)
                        raise RuntimeError(
                            "Out of memory during training. "
                            f"train_batch_size={train_batch_size}, grad_accum_steps={grad_accum_steps}, "
                            f"effective_batch_size={effective_batch_size}, epoch={epoch}, batch={batch_idx}."
                        ) from exc
                    raise

                if batch_idx % grad_accum_steps == 0 or batch_idx == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                batches += 1

            train_loss = running_loss / max(batches, 1)
            try:
                val_metrics, val_loss = evaluate_model(
                    model, val_loader, device, threshold, segment_cfg, float(config.get("fps", 15.0)), criterion
                )
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    _clear_device_cache(device)
                    raise RuntimeError(
                        "Out of memory during evaluation. "
                        f"eval_batch_size={eval_batch_size}, epoch={epoch}."
                    ) from exc
                raise

            log_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_row) + "\n")
            if wandb_run is not None:
                wandb_run.log(log_row, step=epoch)

            metric_value = float(val_metrics.get("balanced_accuracy", 0.0))
            if metric_value > (best_metric + early_stopping_min_delta):
                best_metric = metric_value
                epochs_without_improvement = 0
                _save_checkpoint(run_dir / "checkpoints" / "best.pth", model, optimizer, epoch, log_row)
            else:
                epochs_without_improvement += 1
            _save_checkpoint(run_dir / "checkpoints" / "last.pth", model, optimizer, epoch, log_row)
            if config.get("save_every_n") and epoch % int(config.get("save_every_n")) == 0:
                _save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch}.pth", model, optimizer, epoch, log_row)

            logger.info("Epoch %s: train_loss=%.4f val_loss=%.4f bal_acc=%.4f", epoch, train_loss, val_loss, metric_value)
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %s after %s epochs without validation balanced_accuracy improvement > %.4f",
                    epoch,
                    epochs_without_improvement,
                    early_stopping_min_delta,
                )
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, Any]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        str(path),
    )


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        requested = str(device).lower()
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("Requested device 'mps' is unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message


def _clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _maybe_init_wandb(config: Dict[str, Any], run_dir: Path):
    wandb_cfg = config.get("wandb") if isinstance(config.get("wandb"), dict) else {}
    if not wandb_cfg or not bool(wandb_cfg.get("enabled", False)):
        return None
    project = wandb_cfg.get("project")
    if not project:
        logger.warning("wandb.enabled=true but wandb.project is not set; skipping wandb logging")
        return None

    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to import wandb: %s", exc)
        return None

    init_kwargs = {
        "project": project,
        "entity": wandb_cfg.get("entity"),
        "name": str(config.get("run_id") or run_dir.name),
        "config": _to_jsonable(config),
        "reinit": True,
    }
    if wandb_cfg.get("group"):
        init_kwargs["group"] = str(wandb_cfg.get("group"))
    tags = wandb_cfg.get("tags")
    if isinstance(tags, (list, tuple)) and tags:
        init_kwargs["tags"] = [str(t) for t in tags]

    run = wandb.init(**init_kwargs)
    return run
