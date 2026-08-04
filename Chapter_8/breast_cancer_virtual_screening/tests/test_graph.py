from __future__ import annotations

import torch

from bcvs.models.gnn import GINRegressor
from bcvs.models.graph import collate_graphs, smiles_to_graph


def test_graph_batch_and_forward() -> None:
    batch = collate_graphs([smiles_to_graph("CCO", 6.0, 0), smiles_to_graph("c1ccccc1", 7.0, 1)])
    model = GINRegressor(hidden_dim=32, num_layers=2, dropout=0.0)
    output = model(batch["atom_features"], batch["edge_index"], batch["edge_features"], batch["batch_index"])
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
