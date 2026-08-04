from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bcvs.data.splits import scaffold_split
from bcvs.utils import atomic_write_json


def fingerprint_matrix(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        fp = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, matrix[idx])
    return matrix


def train_extra_trees_baseline(
    table_path: str | Path,
    output_dir: str | Path,
    training_cfg: dict[str, Any],
    seed: int = 2026,
) -> dict[str, Any]:
    table = pd.read_csv(table_path)
    smiles = table["standardized_smiles"].astype(str).tolist()
    y = table["pIC50"].astype(float).to_numpy()
    train_idx, val_idx, test_idx = scaffold_split(
        smiles,
        training_cfg["train_fraction"],
        training_cfg["validation_fraction"],
        training_cfg["test_fraction"],
        seed,
    )
    x = fingerprint_matrix(smiles)
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x[train_idx], y[train_idx])
    metrics = {}
    for name, indices in {"validation": val_idx, "test": test_idx}.items():
        pred = model.predict(x[indices])
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y[indices], pred))),
            "mae": float(mean_absolute_error(y[indices], pred)),
            "r2": float(r2_score(y[indices], pred)),
            "n": int(len(indices)),
        }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "extra_trees_morgan.joblib")
    atomic_write_json(metrics, output_dir / "extra_trees_metrics.json")
    return metrics
