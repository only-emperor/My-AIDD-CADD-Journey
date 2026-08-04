from __future__ import annotations

import gzip
from pathlib import Path

from bcvs.data.io import infer_smiles_column, iter_candidate_chunks


def test_gzipped_whitespace_smiles(tmp_path: Path) -> None:
    path = tmp_path / "library.smi.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("CCO ethanol\nCCN ethylamine\n")
    chunks = list(iter_candidate_chunks(path, chunk_size=1))
    assert len(chunks) == 2
    assert chunks[0].loc[0, "smiles"] == "CCO"
    assert chunks[0].loc[0, "source_id"] == "ethanol"


def test_infer_standardized_smiles_column() -> None:
    assert infer_smiles_column(["id", "standardized_smiles"]) == "standardized_smiles"
