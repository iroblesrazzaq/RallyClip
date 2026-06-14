"""Training loop for the per-frame segment head (E2ESegLoss).

Mirrors training.train.loop but with the 3-output model, the e2e segment loss,
and the stitched six-bin evaluation. Checkpoint selection/early stopping is
configurable: "val_loss" (minimize) or "acceptable" (maximize the six-bin
good+decent share). Writes a full manifest.json into the run dir at the end."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from training.dataset.hdf5_dataset import Hdf5SequenceDataset
from training.eval.seg_evaluator import DecodeConfig, evaluate_seg_model
from training.metrics.segments6 import SixBinConfig
from training.models.seg_lstm import TennisPointSegLSTM
from training.train.seg_loss import E2ESegLoss, SegLossConfig

logger = logging.getLogger(__name__)


def train_seg(dataset_dir: Path, run_dir: Path, config: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.h5"
    val_path = dataset_dir / "val.h5"
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val dataset not found: {val_path}")

    train_ds = Hdf5SequenceDataset(train_path)

    device = _resolve_device(config.get("device"))
    head = str(config.get("head", "linear"))
    model = TennisPointSegLSTM(input_size=train_ds.feature_dim, head=head).to(device)

    loss_cfg = SegLossConfig(
        fps=float(config.get("fps", 5.0)),
        pos_weight=float(config.get("pos_weight", 3.0)),
        cls_weight=float(config.get("cls_weight", 1.0)),
        boundary_weight=float(config.get("boundary_weight", 1.0)),
        diou_weight=float(config.get("diou_weight", 0.25)),
        lambda0=float(config.get("lambda0", 0.05)),
        lambda1=float(config.get("lambda1", 0.2)),
        lambda2=float(config.get("lambda2", 1.0)),
        mu0=float(config.get("mu0", 0.05)),
        mu1=float(config.get("mu1", 0.4)),
        mu2=float(config.get("mu2", 2.0)),
    )
    criterion = E2ESegLoss(loss_cfg).to(device)
    criterion_cpu = E2ESegLoss(loss_cfg)  # for CPU-side artifact eval (see eval block)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 0.01)
    )

    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    decode_cfg = DecodeConfig(
        threshold=float(config.get("threshold", 0.5)),
        vote=str(config.get("decode_vote", "mean")),
    )
    six_bin_cfg = SixBinConfig()
    early_stopping_patience = max(0, int(config.get("early_stopping_patience", 0)))
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))
    selection_metric = str(config.get("selection_metric", "val_loss"))
    if selection_metric not in ("val_loss", "acceptable", "good_weighted"):
        raise ValueError(f"Unknown selection_metric: {selection_metric}")

    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []
    started_at = time.time()

    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not config_path.exists():
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable({**config, "loss": asdict(loss_cfg)}), handle, indent=2)

    logger.info("Seg training setup: device=%s batch=%d loss=%s", device, batch_size, loss_cfg)

    for epoch in range(1, int(config.get("epochs", 30)) + 1):
        epoch_started = time.time()
        model.train()
        running = {"loss": 0.0, "loss_cls": 0.0, "loss_start": 0.0, "loss_end": 0.0, "loss_diou": 0.0}
        batches = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, d_start, d_end = model(features)
            loss, components = criterion(logits, d_start, d_end, targets)
            loss.backward()
            optimizer.step()
            running["loss"] += float(loss.item())
            for key, value in components.items():
                running[key] += value
            batches += 1

        train_means = {f"train_{k}": v / max(batches, 1) for k, v in running.items()}
        # Evaluate a reloaded copy on CPU, never the live MPS model. The MPS LSTM's
        # forward-effective flat weights fork from the registered Parameters during
        # training, and — critically — even a fresh model reloaded from the saved
        # CPU state still evaluates wrong on MPS while the live model coexists in the
        # same process (confirmed: same weights give good 0.175 on CPU vs 0.348 on
        # MPS in-loop). CPU eval of the reloaded weights is exactly what ships, so
        # selection and logging rank the real on-disk artifact.
        cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        eval_model = TennisPointSegLSTM(input_size=train_ds.feature_dim, head=head)
        eval_model.load_state_dict(cpu_state)
        eval_model.eval()
        val_metrics, val_loss = evaluate_seg_model(
            eval_model, val_path, torch.device("cpu"), criterion_cpu, decode_cfg, six_bin_cfg, batch_size=batch_size
        )
        del eval_model

        log_row = {
            "epoch": epoch,
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

        if selection_metric == "val_loss":
            score = -val_loss
        elif selection_metric == "good_weighted":
            score = 2.0 * float(val_metrics.get("share_good", 0.0)) + float(val_metrics.get("share_decent", 0.0))
        else:
            score = float(val_metrics.get("share_good", 0.0)) + float(val_metrics.get("share_decent", 0.0))

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
        loss_cfg=loss_cfg,
        history=history,
        best_epoch=best_epoch,
        selection_metric=selection_metric,
        total_seconds=time.time() - started_at,
        feature_dim=train_ds.feature_dim,
    )


def write_run_manifest(
    *,
    run_dir: Path,
    dataset_dir: Path,
    config: Dict[str, Any],
    loss_cfg,
    history: List[Dict[str, Any]],
    best_epoch: int,
    selection_metric: str,
    total_seconds: float,
    feature_dim: int,
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
            "model_type": "e2e_segment",
        },
        "model": {
            "architecture": "TennisPointSegLSTM",
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "dropout": 0.2,
            "input_size": feature_dim,
            "outputs": ["pointness_logit", "d_start_seconds", "d_end_seconds"],
        },
        "loss": _to_jsonable(asdict(loss_cfg)),
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
        "decode": {
            "method": "offset_vote",
            "threshold": float(config.get("threshold", 0.5)),
            "vote": str(config.get("decode_vote", "mean")),
            "description": "sigmoid pointness >= threshold gates voters; segment = prob-weighted mean of (t - d_start, t + d_end); overlapping segments merged",
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


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, Any]) -> None:
    # torch.save of MPS-resident tensors has been observed to write corrupted
    # weights for this model (file re-evals far below the live model, while
    # explicit .cpu() copies are faithful). Always serialize CPU clones and
    # verify the written file before trusting it.
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "metrics": metrics,
        },
        str(path),
    )
    written = torch.load(str(path), map_location="cpu")["model_state_dict"]
    for key, value in model_state.items():
        if not torch.equal(written[key], value):
            raise RuntimeError(f"Checkpoint verification failed for {path} (tensor {key})")


def _resolve_device(device) -> torch.device:
    if device:
        requested = str(device).lower()
        if requested == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("Requested device 'mps' is unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device(requested)
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
