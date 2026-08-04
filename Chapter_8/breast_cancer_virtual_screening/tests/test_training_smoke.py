from __future__ import annotations

from pathlib import Path

import pandas as pd

from bcvs.models.train import train_single_model


def test_tiny_training_run(tmp_path: Path) -> None:
    smiles = [
        "CCO", "CCN", "CCC", "CCCO", "CCCN", "CCCC", "CCOC", "CCNC",
        "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1",
        "c1ccncc1", "Cc1ccncc1", "C1CCCCC1", "CC1CCCCC1",
        "CC(=O)O", "CC(=O)N", "COC", "CNC",
    ]
    table = pd.DataFrame(
        {
            "standardized_smiles": smiles,
            "pIC50": [5.0 + 0.08 * i for i in range(len(smiles))],
        }
    )
    cfg = {
        "train_fraction": 0.7,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "batch_size": 8,
        "epochs": 2,
        "patience": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.0,
        "minimum_delta": 0.0,
        "device": "cpu",
        "target_normalization": True,
    }
    result = train_single_model(table, cfg, seed=13, output_dir=tmp_path)
    assert Path(result["checkpoint"]).exists()
    assert result["validation_rmse"] >= 0
