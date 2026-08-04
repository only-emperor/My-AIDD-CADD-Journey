from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_for_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or f"ACYCLIC:{Chem.MolToSmiles(mol, canonical=True)}"


def scaffold_split(
    smiles: Sequence[str],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to 1.0")
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles):
        groups[scaffold_for_smiles(smi)].append(idx)
    rng = random.Random(seed)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)

    n_total = len(smiles)
    target_train = train_fraction * n_total
    target_val = validation_fraction * n_total
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for group in grouped:
        if len(train) + len(group) <= target_train or len(train) < target_train * 0.95:
            train.extend(group)
        elif len(val) + len(group) <= target_val or len(val) < target_val * 0.95:
            val.extend(group)
        else:
            test.extend(group)
    if not val or not test:
        # Deterministic fallback for very small/scaffold-poor datasets.
        all_idx = list(range(n_total))
        rng.shuffle(all_idx)
        n_train = max(1, int(train_fraction * n_total))
        n_val = max(1, int(validation_fraction * n_total))
        train = all_idx[:n_train]
        val = all_idx[n_train : n_train + n_val]
        test = all_idx[n_train + n_val :]
    return np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test))
