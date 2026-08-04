from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from bcvs.models.gnn import GINRegressor
from bcvs.models.graph import MoleculeGraphDataset, collate_graphs
from bcvs.utils import select_device


@dataclass
class LoadedMember:
    model: GINRegressor
    weight: float
    target_mean: float
    target_std: float
    seed: int


class GNNEnsemble:
    def __init__(self, manifest_path: str | Path, device: str = "auto"):
        manifest_path = Path(manifest_path)
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.target_symbol = self.manifest["target_symbol"]
        self.device = select_device(device)
        self.members: list[LoadedMember] = []
        for member in self.manifest["members"]:
            checkpoint_path = Path(member["checkpoint"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = (manifest_path.parent / checkpoint_path).resolve()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = GINRegressor(**checkpoint["model_kwargs"])
            model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device).eval()
            self.members.append(
                LoadedMember(
                    model=model,
                    weight=float(member["weight"]),
                    target_mean=float(checkpoint["target_mean"]),
                    target_std=float(checkpoint["target_std"]),
                    seed=int(checkpoint["seed"]),
                )
            )

    def predict(self, smiles: Iterable[str], batch_size: int = 256) -> dict[str, np.ndarray]:
        smiles_list = list(smiles)
        dataset = MoleculeGraphDataset(smiles_list)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
        member_predictions = []
        for member in self.members:
            preds = []
            with torch.inference_mode():
                for batch in loader:
                    atom_features = batch["atom_features"].to(self.device)
                    edge_index = batch["edge_index"].to(self.device)
                    edge_features = batch["edge_features"].to(self.device)
                    batch_index = batch["batch_index"].to(self.device)
                    out = member.model(atom_features, edge_index, edge_features, batch_index)
                    out = out * member.target_std + member.target_mean
                    preds.append(out.detach().cpu().numpy())
            member_predictions.append(np.concatenate(preds) if preds else np.array([]))
        matrix = np.vstack(member_predictions)
        weights = np.array([m.weight for m in self.members], dtype=float)
        mean = np.average(matrix, axis=0, weights=weights)
        variance = np.average((matrix - mean) ** 2, axis=0, weights=weights)
        return {
            "mean": mean,
            "std": np.sqrt(np.maximum(variance, 0.0)),
            "members": matrix,
            "weights": weights,
        }
