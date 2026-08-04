from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from bcvs.chem.descriptors import molecular_descriptors


@dataclass
class FilterResult:
    pass_filter: bool
    reasons: list[str]
    annotations: dict[str, Any]


class StructuralAlertFilter:
    def __init__(self) -> None:
        self.catalogs: dict[str, FilterCatalog] = {}
        mapping = {
            "pains": FilterCatalogParams.FilterCatalogs.PAINS,
            "brenk": FilterCatalogParams.FilterCatalogs.BRENK,
            "nih": FilterCatalogParams.FilterCatalogs.NIH,
        }
        for name, catalog_type in mapping.items():
            params = FilterCatalogParams()
            params.AddCatalog(catalog_type)
            self.catalogs[name] = FilterCatalog(params)

    def annotate(self, smiles: str) -> dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid standardized SMILES")
        result: dict[str, Any] = {}
        for name, catalog in self.catalogs.items():
            matches = catalog.GetMatches(mol)
            labels = [match.GetDescription() for match in matches]
            result[f"{name}_count"] = len(labels)
            result[f"{name}_alerts"] = ";".join(labels)
        return result


def apply_physchem_filter(smiles: str, cfg: dict[str, Any]) -> FilterResult:
    d = molecular_descriptors(smiles)
    reasons: list[str] = []

    def outside(value: float, bounds: list[float]) -> bool:
        return value < bounds[0] or value > bounds[1]

    if outside(d["mw"], cfg["molecular_weight"]):
        reasons.append("molecular_weight")
    if outside(d["logp"], cfg["logp"]):
        reasons.append("logp")
    if outside(d["tpsa"], cfg["tpsa"]):
        reasons.append("tpsa")
    if d["hbd"] > cfg["hbd_max"]:
        reasons.append("hbd")
    if d["hba"] > cfg["hba_max"]:
        reasons.append("hba")
    if d["rotatable_bonds"] > cfg["rotatable_bonds_max"]:
        reasons.append("rotatable_bonds")
    if abs(d["formal_charge"]) > cfg["formal_charge_abs_max"]:
        reasons.append("formal_charge")
    if d["lipinski_violations"] > cfg["lipinski_violations_max"]:
        reasons.append("lipinski_violations")
    if "synthetic_accessibility_max" in cfg and np.isfinite(d["sa_score"]):
        if d["sa_score"] > float(cfg["synthetic_accessibility_max"]):
            reasons.append("synthetic_accessibility")
    return FilterResult(not reasons, reasons, d)


OPS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt}


def evaluate_admet_rules(
    row: dict[str, Any], rules: dict[str, dict[str, Any]]
) -> tuple[bool, list[str], float, int]:
    hard_pass = True
    reasons: list[str] = []
    desirabilities: list[float] = []
    matched = 0
    normalized_keys = {str(key).lower(): str(key) for key in row}
    for endpoint, rule in rules.items():
        actual = endpoint if endpoint in row else normalized_keys.get(endpoint.lower())
        if actual is None:
            prefix_matches = [
                original for lowered, original in normalized_keys.items()
                if lowered.startswith(endpoint.lower() + "_")
            ]
            actual = sorted(prefix_matches)[0] if prefix_matches else None
        if actual is None or row.get(actual) is None:
            continue
        try:
            value = float(row[actual])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        matched += 1
        op_text = str(rule["operator"])
        threshold = float(rule["threshold"])
        passed = OPS[op_text](value, threshold)
        if not passed and bool(rule.get("hard", False)):
            hard_pass = False
            reasons.append(f"admet:{actual}")
        if op_text in {">=", ">"}:
            desirability = 1.0 / (1.0 + np.exp(-8.0 * (value - threshold)))
        else:
            desirability = 1.0 / (1.0 + np.exp(8.0 * (value - threshold)))
        desirabilities.append(float(desirability))
    score = float(np.mean(desirabilities)) if desirabilities else float("nan")
    return hard_pass, reasons, score, matched
