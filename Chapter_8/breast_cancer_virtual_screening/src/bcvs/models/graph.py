from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from rdkit import Chem
from torch.utils.data import Dataset

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
    Chem.rdchem.HybridizationType.S: 1,
    Chem.rdchem.HybridizationType.SP: 2,
    Chem.rdchem.HybridizationType.SP2: 3,
    Chem.rdchem.HybridizationType.SP3: 4,
    Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6,
    Chem.rdchem.HybridizationType.OTHER: 7,
}
CHIRAL_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}
BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
BOND_STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}

ATOM_CARDINALITIES = [119, 11, 11, 8, 2, 9, 4]
BOND_CARDINALITIES = [5, 2, 2, 6]


@dataclass
class GraphSample:
    atom_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    y: torch.Tensor | None
    smiles: str
    row_index: int


def smiles_to_graph(smiles: str, y: float | None = None, row_index: int = -1) -> GraphSample:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    features = []
    for atom in mol.GetAtoms():
        atomic_num = min(max(atom.GetAtomicNum(), 0), 118)
        degree = min(atom.GetTotalDegree(), 10)
        formal_charge = min(max(atom.GetFormalCharge(), -5), 5) + 5
        hybridization = HYBRIDIZATION_MAP.get(atom.GetHybridization(), 0)
        aromatic = int(atom.GetIsAromatic())
        total_h = min(atom.GetTotalNumHs(includeNeighbors=True), 8)
        chirality = CHIRAL_MAP.get(atom.GetChiralTag(), 0)
        features.append(
            [atomic_num, degree, formal_charge, hybridization, aromatic, total_h, chirality]
        )
    edges: list[list[int]] = [[], []]
    edge_features: list[list[int]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges[0].extend([begin, end])
        edges[1].extend([end, begin])
        bond_feature = [
            BOND_TYPE_MAP.get(bond.GetBondType(), 4),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            BOND_STEREO_MAP.get(bond.GetStereo(), 0),
        ]
        edge_features.extend([bond_feature, bond_feature])
    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_tensor = (
        torch.tensor(edge_features, dtype=torch.long)
        if edge_features
        else torch.empty((0, len(BOND_CARDINALITIES)), dtype=torch.long)
    )
    atom_tensor = torch.tensor(features, dtype=torch.long)
    target = None if y is None else torch.tensor(float(y), dtype=torch.float32)
    return GraphSample(atom_tensor, edge_index, edge_tensor, target, smiles, row_index)


class MoleculeGraphDataset(Dataset[GraphSample]):
    def __init__(self, smiles: Sequence[str], targets: Sequence[float] | None = None):
        self.smiles = list(smiles)
        self.targets = None if targets is None else list(targets)
        if self.targets is not None and len(self.smiles) != len(self.targets):
            raise ValueError("SMILES and targets must have the same length")

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> GraphSample:
        target = None if self.targets is None else self.targets[index]
        return smiles_to_graph(self.smiles[index], target, index)


def collate_graphs(samples: list[GraphSample]) -> dict[str, torch.Tensor | list[str]]:
    atom_features = []
    edge_indices = []
    all_edge_features = []
    batch_index = []
    targets = []
    smiles = []
    row_indices = []
    offset = 0
    has_targets = all(sample.y is not None for sample in samples)
    for graph_idx, sample in enumerate(samples):
        atom_features.append(sample.atom_features)
        if sample.edge_index.numel() > 0:
            edge_indices.append(sample.edge_index + offset)
            all_edge_features.append(sample.edge_features)
        batch_index.append(torch.full((sample.atom_features.shape[0],), graph_idx, dtype=torch.long))
        if has_targets:
            targets.append(sample.y)
        smiles.append(sample.smiles)
        row_indices.append(sample.row_index)
        offset += sample.atom_features.shape[0]
    edges = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_features = (
        torch.cat(all_edge_features, dim=0)
        if all_edge_features
        else torch.empty((0, len(BOND_CARDINALITIES)), dtype=torch.long)
    )
    batch: dict[str, torch.Tensor | list[str]] = {
        "atom_features": torch.cat(atom_features, dim=0),
        "edge_index": edges,
        "edge_features": edge_features,
        "batch_index": torch.cat(batch_index, dim=0),
        "smiles": smiles,
        "row_indices": torch.tensor(row_indices, dtype=torch.long),
    }
    if has_targets:
        batch["y"] = torch.stack([x for x in targets if x is not None])
    return batch
