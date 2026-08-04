from __future__ import annotations

import torch
from torch import nn

from bcvs.models.graph import ATOM_CARDINALITIES, BOND_CARDINALITIES


class CategoricalFeatureEncoder(nn.Module):
    def __init__(self, cardinalities: list[int], hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, hidden_dim) for cardinality in cardinalities]
        )
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] == 0:
            return torch.empty((0, self.embeddings[0].embedding_dim), device=features.device)
        encoded = self.embeddings[0](features[:, 0])
        for idx, embedding in enumerate(self.embeddings[1:], start=1):
            encoded = encoded + embedding(features[:, idx])
        return encoded


class GINELayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_encoder = CategoricalFeatureEncoder(BOND_CARDINALITIES, hidden_dim)
        self.message_activation = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        aggregated = torch.zeros_like(x)
        if edge_index.numel() > 0:
            source, target = edge_index[0], edge_index[1]
            edge_embedding = self.edge_encoder(edge_features)
            messages = self.message_activation(x[source] + edge_embedding)
            aggregated.index_add_(0, target, messages)
        updated = self.mlp((1.0 + self.eps) * x + aggregated)
        return self.norm(x + self.dropout(updated))


class GINRegressor(nn.Module):
    """Edge-aware GINE-style molecular graph regressor."""

    def __init__(self, hidden_dim: int = 192, num_layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.atom_encoder = CategoricalFeatureEncoder(ATOM_CARDINALITIES, hidden_dim)
        self.layers = nn.ModuleList([GINELayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def global_mean_pool(x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        n_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        pooled = torch.zeros((n_graphs, x.shape[1]), dtype=x.dtype, device=x.device)
        pooled.index_add_(0, batch_index, x)
        counts = torch.bincount(batch_index, minlength=n_graphs).clamp(min=1).to(x.dtype).unsqueeze(1)
        return pooled / counts

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.atom_encoder(atom_features)
        for layer in self.layers:
            x = layer(x, edge_index, edge_features)
        graph_embeddings = self.global_mean_pool(x, batch_index)
        return self.head(graph_embeddings).squeeze(-1)
