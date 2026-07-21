"""Training loop for the boundary-heatmap head (E2EHeatmapLoss).

Mirrors training.train.seg_loop: same 3-output model shape, the heatmap loss,
and the stitched six-bin evaluation via a CPU-reloaded copy of the model (the
MPS LSTM eval-divergence workaround is device-level, so it applies here too).
Checkpoint selection/early stopping match seg_loop ("val_loss" | "acceptable" |
"good_weighted"). Writes a manifest.json into the run dir at the end."""
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
from training.eval.heatmap_evaluator import HeatmapDecodeConfig, evaluate_heatmap_model
from training.metrics.segments6 import SixBinConfig
from training.models.heatmap_gru import TennisPointHeatmapGRU
from training.models.heatmap_lstm import TennisPointHeatmapLSTM
from training.train.heatmap_loss import E2EHeatmapLoss, HeatmapLossConfig

logger = logging.getLogger(__name__)


def build_heatmap_model(backbone: str, input_size: int, head: str) -> torch.nn.Module:
    """Backbone selector for the heatmap head. lstm (default) | gru."""
    b = str(backbone).lower()
    if b == "lstm":
        return TennisPointHeatmapLSTM(input_size=input_size, head=head)
    if b == "gru":
        return TennisPointHeatmapGRU(input_size=input_size, head=head)
    raise ValueError(f"Unknown heatmap_backbone: {backbone!r} (expected lstm | gru)")


def _opt_int(config: Dict[str, Any], key: str):
    v = config.get(key)
    return None if v is None else int(v)


def _opt_float(config: Dict[str, Any], key: str):
    v = config.get(key)
    return None if v is None else float(v)


def train_heatmap(dataset_dir: Path, run_dir: Path, config: Dict[str, Any]) -> None:
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
    head = str(config.get("heatmap_head", "mlp"))
    backbone = str(config.get("heatmap_backbone", "lstm"))
    model = build_heatmap_model(backbone, train_ds.feature_dim, head).to(device)

    if config.get("pos_weight") is None:
        from training.train.loop import _default_pos_weight

        pos_weight_value = _default_pos_weight(train_ds)
        logger.info("Derived pos_weight=%.4f from train positive rate", pos_weight_value)
    else:
        pos_weight_value = float(config.get("pos_weight"))

    fps = float(config.get("fps", 5.0))
    sigma_seconds = float(config.get("heatmap_sigma_seconds", 0.5))
    loss_cfg = HeatmapLossConfig(
        fps=fps,
        sigma_seconds=sigma_seconds,
        pos_weight=pos_weight_value,
        cls_weight=float(config.get("heatmap_cls_weight", 1.0)),
        start_weight=float(config.get("heatmap_start_weight", 1.0)),
        end_weight=float(config.get("heatmap_end_weight", 1.0)),
        heatmap_loss=str(config.get("heatmap_loss", "bce")),
        heatmap_pos_threshold=float(config.get("heatmap_pos_threshold", 0.1)),
        heatmap_pos_weight=float(config.get("heatmap_pos_weight", 20.0)),
        focal_alpha=float(config.get("heatmap_focal_alpha", 2.0)),
        focal_beta=float(config.get("heatmap_focal_beta", 4.0)),
    )
    criterion = E2EHeatmapLoss(loss_cfg).to(device)
    criterion_cpu = E2EHeatmapLoss(loss_cfg)  # for CPU-side artifact eval (see eval block)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 0.01)
    )

    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    decode_cfg = HeatmapDecodeConfig(
        mode=str(config.get("heatmap_decode_mode", "hybrid")),
        threshold=float(config.get("threshold", 0.5)),
        peak_threshold=float(config.get("heatmap_peak_threshold", 0.3)),
        sigma_frames=sigma_seconds * fps,
        refine_window_frames=_opt_int(config, "heatmap_decode_window_frames"),
        nms_frames=_opt_int(config, "heatmap_nms_frames"),
        min_duration_sec=float(config.get("heatmap_min_duration_sec", 0.3)),
        max_duration_sec=float(config.get("heatmap_max_duration_sec", 60.0)),
        pointness_gate=_opt_float(config, "heatmap_pointness_gate"),
    )
    six_bin_cfg = SixBinConfig()
    early_stopping_patience = max(0, int(config.get("early_stopping_patience", 0)))
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))
    selection_metric = str(config.get("selection_metric", "acceptable"))
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

    logger.info("Heatmap training setup: device=%s batch=%d loss=%s", device, batch_size, loss_cfg)

    for epoch in range(1, int(config.get("epochs", 30)) + 1):
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
        # Evaluate a reloaded copy on CPU, never the live MPS model — the MPS LSTM's
        # forward-effective flat weights fork from the registered Parameters during
        # training and even a fresh reloaded model evaluates wrong on MPS while the
        # live model coexists in-process (documented at length in seg_loop.py). CPU
        # eval of the reloaded weights is exactly what ships.
        cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        eval_model = build_heatmap_model(backbone, train_ds.feature_dim, head)
        eval_model.load_state_dict(cpu_state)
        eval_model.eval()
        val_metrics, val_loss = evaluate_heatmap_model(
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
        decode_cfg=decode_cfg,
        history=history,
        best_epoch=best_epoch,
        selection_metric=selection_metric,
        total_seconds=time.time() - started_at,
        feature_dim=train_ds.feature_dim,
        backbone=backbone,
    )


def write_run_manifest(
    *,
    run_dir: Path,
    dataset_dir: Path,
    config: Dict[str, Any],
    loss_cfg,
    decode_cfg,
    history: List[Dict[str, Any]],
    best_epoch: int,
    selection_metric: str,
    total_seconds: float,
    feature_dim: int,
    backbone: str = "lstm",
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
            "architecture": "TennisPointHeatmapGRU" if str(backbone).lower() == "gru" else "TennisPointHeatmapLSTM",
            "backbone": str(backbone).lower(),
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "dropout": 0.2,
            "input_size": feature_dim,
            "outputs": ["pointness_logit", "start_heatmap_logit", "end_heatmap_logit"],
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
            "method": "gaussian_peak_softargmax",
            "mode": decode_cfg.mode,
            "threshold": decode_cfg.threshold,
            "peak_threshold": decode_cfg.peak_threshold,
            "description": (
                "hybrid: pointness runs define segments, start/end heatmaps refine each "
                "edge via soft-argmax; peakpair: peak-pick + greedy pair. Overlaps merged."
            ),
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
    # Serialize CPU clones and verify the written file — torch.save of MPS-resident
    # tensors has been observed to write corrupted weights for this LSTM family
    # (see seg_loop.py).
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
