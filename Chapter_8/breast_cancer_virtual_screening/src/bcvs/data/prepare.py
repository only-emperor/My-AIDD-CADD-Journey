from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bcvs.chem.standardize import MoleculeStandardizer
from bcvs.utils import write_dataframe

LOGGER = logging.getLogger(__name__)


def standardize_activity_table(
    activities: pd.DataFrame,
    standardization_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standardizer = MoleculeStandardizer(standardization_cfg)
    records = [standardizer.standardize(s).to_dict() for s in activities["canonical_smiles"]]
    std = pd.DataFrame(records)
    out = pd.concat([activities.reset_index(drop=True), std], axis=1)
    invalid = out[~out["valid"]].copy()
    valid = out[out["valid"]].copy()
    return valid.reset_index(drop=True), invalid.reset_index(drop=True)


def aggregate_target_activities(
    standardized: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if standardized.empty:
        return standardized.copy(), pd.DataFrame()
    agg_mode = cfg.get("aggregate", "median")
    grouped = standardized.groupby(["target_symbol", "standardized_smiles", "inchikey"], dropna=False)
    rows = []
    excluded = []
    for keys, group in grouped:
        values = group["pIC50"].astype(float)
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row = {
            "target_symbol": keys[0],
            "standardized_smiles": keys[1],
            "inchikey": keys[2],
            "pIC50": float(values.median() if agg_mode == "median" else values.mean()),
            "replicate_count": int(len(values)),
            "replicate_pic50_sd": sd,
            "pic50_min": float(values.min()),
            "pic50_max": float(values.max()),
            "molecule_chembl_ids": ";".join(sorted(set(group["molecule_chembl_id"].dropna().astype(str)))),
            "assay_chembl_ids": ";".join(sorted(set(group["assay_chembl_id"].dropna().astype(str))))
            if "assay_chembl_id" in group
            else "",
        }
        if sd > float(cfg.get("max_replicate_pic50_sd", 1.5)):
            row["exclusion_reason"] = "replicate_variability"
            excluded.append(row)
        else:
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(excluded)


def prepare_all_targets(
    clean_activity_paths: list[Path],
    standardization_cfg: dict[str, Any],
    chembl_cfg: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for path in clean_activity_paths:
        raw = pd.read_csv(path)
        if raw.empty:
            continue
        symbol = str(raw["target_symbol"].iloc[0])
        valid, invalid = standardize_activity_table(raw, standardization_cfg)
        aggregated, excluded = aggregate_target_activities(valid, chembl_cfg)
        outputs[symbol] = write_dataframe(aggregated, output_dir / f"training_{symbol}.csv.gz")
        write_dataframe(invalid, output_dir / f"invalid_structures_{symbol}.csv.gz")
        write_dataframe(excluded, output_dir / f"excluded_aggregates_{symbol}.csv.gz")
        LOGGER.info("Prepared %s: %d aggregated molecules", symbol, len(aggregated))
    return outputs
