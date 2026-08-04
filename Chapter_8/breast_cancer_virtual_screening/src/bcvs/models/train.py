from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from bcvs.data.splits import scaffold_split
from bcvs.models.gnn import GINRegressor
from bcvs.models.graph import MoleculeGraphDataset, collate_graphs
from bcvs.utils import atomic_write_json, seed_everything, select_device, write_dataframe

LOGGER = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    n: int


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    return RegressionMetrics(
        rmse=float(math.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        n=int(len(y_true)),
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            output = model(batch["atom_features"], batch["edge_index"], batch["edge_features"], batch["batch_index"])
            output = output * target_std + target_mean
            preds.append(output.detach().cpu().numpy())
            row_indices.append(batch["row_indices"].detach().cpu().numpy())
            if "y" in batch:
                truth.append(batch["y"].detach().cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(truth) if truth else np.array([])
    indices = np.concatenate(row_indices) if row_indices else np.array([], dtype=int)
    return y_true, y_pred, indices


def train_single_model(
    table: pd.DataFrame,
    training_cfg: dict[str, Any],
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(training_cfg.get("device", "auto"))
    smiles = table["standardized_smiles"].astype(str).tolist()
    targets = table["pIC50"].astype(float).to_numpy()
    train_idx, val_idx, test_idx = scaffold_split(
        smiles,
        training_cfg["train_fraction"],
        training_cfg["validation_fraction"],
        training_cfg["test_fraction"],
        seed,
    )
    target_mean = float(np.mean(targets[train_idx])) if training_cfg.get("target_normalization", True) else 0.0
    target_std = float(np.std(targets[train_idx])) if training_cfg.get("target_normalization", True) else 1.0
    target_std = max(target_std, 1e-6)
    normalized_targets = (targets - target_mean) / target_std
    dataset = MoleculeGraphDataset(smiles, normalized_targets)
    loader_args = {
        "batch_size": int(training_cfg["batch_size"]),
        "collate_fn": collate_graphs,
        "num_workers": 0,
    }
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), shuffle=False, **loader_args)
    test_loader = DataLoader(Subset(dataset, test_idx.tolist()), shuffle=False, **loader_args)

    model_kwargs = {
        "hidden_dim": int(training_cfg["hidden_dim"]),
        "num_layers": int(training_cfg["num_layers"]),
        "dropout": float(training_cfg["dropout"]),
    }
    model = GINRegressor(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        losses = []
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["atom_features"], batch["edge_index"], batch["edge_features"], batch["batch_index"])
            loss = loss_fn(pred, batch["y"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        y_val_norm, pred_val, _ = predict_loader(model, val_loader, device, target_mean, target_std)
        y_val = y_val_norm * target_std + target_mean
        val_rmse = regression_metrics(y_val, pred_val).rmse
        scheduler.step(val_rmse)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "validation_rmse": val_rmse,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if val_rmse < best_val_rmse - float(training_cfg.get("minimum_delta", 0.0005)):
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= int(training_cfg["patience"]):
            break

    model.load_state_dict(best_state)
    split_rows = []
    prediction_frames = []
    split_map = {"train": train_idx, "validation": val_idx, "test": test_idx}
    metrics: dict[str, Any] = {}
    for split_name, indices in split_map.items():
        loader = DataLoader(Subset(dataset, indices.tolist()), shuffle=False, **loader_args)
        y_norm, pred, original_indices = predict_loader(model, loader, device, target_mean, target_std)
        y_true = y_norm * target_std + target_mean
        metrics[split_name] = asdict(regression_metrics(y_true, pred))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": original_indices,
                    "split": split_name,
                    "observed_pic50": y_true,
                    "predicted_pic50": pred,
                    "residual": y_true - pred,
                }
            )
        )
        split_rows.extend({"row_index": int(i), "split": split_name} for i in indices)

    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
        "model_kwargs": model_kwargs,
        "target_mean": target_mean,
        "target_std": target_std,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_rmse": best_val_rmse,
        "training_config": training_cfg,
    }
    checkpoint_path = output_dir / f"model_seed_{seed}.pt"
    torch.save(checkpoint, checkpoint_path)
    write_dataframe(pd.DataFrame(history), output_dir / f"history_seed_{seed}.csv")
    write_dataframe(pd.DataFrame(split_rows), output_dir / f"splits_seed_{seed}.csv")
    write_dataframe(pd.concat(prediction_frames, ignore_index=True), output_dir / f"predictions_seed_{seed}.csv")
    atomic_write_json(metrics, output_dir / f"metrics_seed_{seed}.json")
    LOGGER.info("Seed %d finished: validation RMSE %.4f at epoch %d", seed, best_val_rmse, best_epoch)
    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "validation_rmse": best_val_rmse,
        "metrics": metrics,
    }


def train_ensemble_for_target(
    table_path: str | Path,
    target_symbol: str,
    training_cfg: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    table = pd.read_csv(table_path)
    if table.empty:
        raise ValueError(f"No training rows for {target_symbol}")
    output_dir = Path(output_dir) / target_symbol
    results = [
        train_single_model(table, training_cfg, int(seed), output_dir)
        for seed in training_cfg["seeds"]
    ]
    rmses = np.array([row["validation_rmse"] for row in results], dtype=float)
    temperature = max(float(np.std(rmses)), 0.05)
    logits = -(rmses - rmses.min()) / temperature
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    members = []
    for result, weight in zip(results, weights, strict=True):
        members.append({**result, "weight": float(weight)})
    manifest = {
        "target_symbol": target_symbol,
        "training_table": str(Path(table_path).resolve()),
        "n_samples": int(len(table)),
        "members": members,
        "weight_method": "softmax_negative_validation_rmse",
    }
    atomic_write_json(manifest, output_dir / "ensemble_manifest.json")
    return manifest
