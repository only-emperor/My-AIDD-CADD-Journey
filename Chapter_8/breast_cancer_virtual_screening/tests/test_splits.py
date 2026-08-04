from __future__ import annotations

from bcvs.data.splits import scaffold_for_smiles, scaffold_split


def test_scaffold_split_has_no_scaffold_overlap() -> None:
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "Oc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "C1CCCCC1",
        "CC1CCCCC1",
        "CCO",
        "CCN",
        "CCCC",
        "CCCO",
        "CCCN",
    ]
    train, val, test = scaffold_split(smiles, 0.6, 0.2, 0.2, seed=13)
    sets = []
    for indices in [train, val, test]:
        sets.append({scaffold_for_smiles(smiles[int(i)]) for i in indices})
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
