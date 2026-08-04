from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
from rdkit import Chem


def infer_smiles_column(columns: list[str]) -> str:
    normalized = {c.lower().strip(): c for c in columns}
    for candidate in [
        "standardized_smiles",
        "smiles",
        "canonical_smiles",
        "isomericsmiles",
        "canonicalsmiles",
    ]:
        if candidate in normalized:
            return normalized[candidate]
    raise KeyError(f"No SMILES column found. Available columns: {columns}")


def iter_candidate_chunks(path: str | Path, chunk_size: int = 5000) -> Iterator[pd.DataFrame]:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        yield from pd.read_csv(path, chunksize=chunk_size)
        return
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        yield from pd.read_csv(path, sep="\t", chunksize=chunk_size)
        return
    if any(suffixes.endswith(ext) for ext in [".smi", ".smiles", ".smi.gz", ".smiles.gz"]):
        rows: list[dict[str, str]] = []
        if suffixes.endswith(".gz"):
            import gzip

            handle_context = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            handle_context = path.open("r", encoding="utf-8", errors="replace")
        with handle_context as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Most SMILES files are whitespace-delimited; tabs are not guaranteed.
                fields = line.split(maxsplit=1)
                rows.append(
                    {
                        "smiles": fields[0].strip(),
                        "source_id": fields[1].strip() if len(fields) > 1 else f"row_{idx}",
                    }
                )
                if len(rows) >= chunk_size:
                    yield pd.DataFrame(rows)
                    rows = []
        if rows:
            yield pd.DataFrame(rows)
        return
    if suffixes.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                yield batch.to_pandas()
        except ImportError:
            frame = pd.read_parquet(path)
            for start in range(0, len(frame), chunk_size):
                yield frame.iloc[start : start + chunk_size].copy()
        return
    if suffixes.endswith(".sdf") or suffixes.endswith(".sdf.gz"):
        if suffixes.endswith(".gz"):
            import gzip

            handle = gzip.open(path, "rb")
        else:
            handle = path.open("rb")
        with handle:
            supplier = Chem.ForwardSDMolSupplier(handle, sanitize=True, removeHs=True)
            rows = []
            for idx, mol in enumerate(supplier):
                if mol is None:
                    rows.append({"smiles": None, "source_id": f"sdf_{idx}", "sdf_parse_error": True})
                else:
                    props = mol.GetPropsAsDict()
                    props["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=True)
                    props.setdefault("source_id", mol.GetProp("_Name") if mol.HasProp("_Name") else f"sdf_{idx}")
                    rows.append(props)
                if len(rows) >= chunk_size:
                    yield pd.DataFrame(rows)
                    rows = []
            if rows:
                yield pd.DataFrame(rows)
        return
    raise ValueError(f"Unsupported candidate format: {path}")
