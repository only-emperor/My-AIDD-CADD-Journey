from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from bcvs.http import HTTPConfig, ResilientSession
from bcvs.utils import atomic_write_json, write_dataframe

LOGGER = logging.getLogger(__name__)

UNIT_TO_MOLAR = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "µM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}


def ic50_to_pic50(value: float | str, unit: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"IC50 must be finite and positive, got {value!r}")
    if unit not in UNIT_TO_MOLAR:
        raise ValueError(f"Unsupported IC50 unit: {unit!r}")
    return -math.log10(numeric * UNIT_TO_MOLAR[unit])


class ChEMBLClient:
    def __init__(self, cfg: dict[str, Any], raw_dir: str | Path):
        self.cfg = cfg
        self.base_url = cfg["base_url"].rstrip("/") + "/"
        self.raw_dir = Path(raw_dir) / "chembl"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.http = ResilientSession(
            HTTPConfig(
                timeout_seconds=cfg.get("timeout_seconds", 60),
                max_retries=cfg.get("max_retries", 6),
            )
        )

    def _endpoint(self, resource: str) -> str:
        return urljoin(self.base_url, f"{resource}.json")

    def _paginate(
        self, resource: str, params: dict[str, Any], cache_prefix: str
    ) -> list[dict[str, Any]]:
        params = dict(params)
        params.setdefault("limit", self.cfg.get("page_size", 1000))
        url: str | None = self._endpoint(resource)
        records: list[dict[str, Any]] = []
        page = 0
        while url:
            payload = self.http.get_json(url, params=params if page == 0 else None)
            atomic_write_json(payload, self.raw_dir / f"{cache_prefix}_page_{page:05d}.json")
            objects = payload.get(f"{resource}s")
            if objects is None:
                # ChEMBL has a few irregular plural names; find the list-valued top-level key.
                objects = next((v for k, v in payload.items() if k != "page_meta" and isinstance(v, list)), [])
            records.extend(objects)
            next_url = payload.get("page_meta", {}).get("next")
            if next_url and next_url.startswith("/"):
                next_url = "https://www.ebi.ac.uk" + next_url
            url = next_url
            page += 1
        return records

    def search_target(self, symbol: str) -> list[dict[str, Any]]:
        url = urljoin(self.base_url, "target/search.json")
        payload = self.http.get_json(url, params={"q": symbol, "limit": 100})
        atomic_write_json(payload, self.raw_dir / f"target_search_{symbol}.json")
        return payload.get("targets", [])

    def resolve_target(
        self,
        symbol: str,
        pinned_id: str | None = None,
        organism: str = "Homo sapiens",
        target_type: str = "SINGLE PROTEIN",
    ) -> dict[str, Any]:
        if pinned_id:
            payload = self.http.get_json(self._endpoint("target").replace(".json", f"/{pinned_id}.json"))
            return payload

        candidates = self.search_target(symbol)
        if not candidates:
            raise LookupError(f"No ChEMBL target found for {symbol}")

        def score(item: dict[str, Any]) -> tuple[int, int, int, int]:
            organism_match = int(item.get("organism") == organism)
            type_match = int(item.get("target_type") == target_type)
            component_synonyms = " ".join(
                str(x)
                for component in item.get("target_components", [])
                for x in component.get("target_component_synonyms", [])
            ).upper()
            text = " ".join(
                [
                    str(item.get("pref_name", "")),
                    str(item.get("target_chembl_id", "")),
                    component_synonyms,
                ]
            ).upper()
            symbol_match = int(symbol.upper() in text)
            component_count_bonus = -abs(len(item.get("target_components", [])) - 1)
            return organism_match, type_match, symbol_match, component_count_bonus

        ranked = sorted(candidates, key=score, reverse=True)
        best = ranked[0]
        if best.get("organism") != organism:
            LOGGER.warning("Best ChEMBL target for %s is not %s: %s", symbol, organism, best)
        return best

    def fetch_ic50_activities(self, target_chembl_id: str, symbol: str) -> pd.DataFrame:
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type": self.cfg.get("activity_type", "IC50"),
        }
        records = self._paginate("activity", params, f"activities_{symbol}_{target_chembl_id}")
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["target_symbol"] = symbol
        df["target_chembl_id"] = target_chembl_id
        return df


def clean_chembl_activities(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned activities and a row-level exclusion audit."""
    if df.empty:
        return df.copy(), pd.DataFrame()
    work = df.copy()
    work["exclusion_reason"] = ""

    def reject(mask: pd.Series, reason: str) -> None:
        eligible = work["exclusion_reason"].eq("") & mask.fillna(False)
        work.loc[eligible, "exclusion_reason"] = reason

    reject(work.get("canonical_smiles", pd.Series(index=work.index)).isna(), "missing_smiles")
    reject(work.get("standard_value", pd.Series(index=work.index)).isna(), "missing_standard_value")
    reject(work.get("standard_units", pd.Series(index=work.index)).isna(), "missing_standard_units")
    reject(work.get("standard_type", "") != cfg.get("activity_type", "IC50"), "wrong_activity_type")

    if cfg.get("only_exact_relation", True) and "standard_relation" in work:
        reject(work["standard_relation"].fillna("").astype(str).str.strip() != "=", "censored_relation")
    if cfg.get("require_standard_flag", True) and "standard_flag" in work:
        reject(pd.to_numeric(work["standard_flag"], errors="coerce") != 1, "nonstandard_record")
    if cfg.get("remove_potential_duplicates", True) and "potential_duplicate" in work:
        reject(pd.to_numeric(work["potential_duplicate"], errors="coerce").fillna(0) != 0, "potential_duplicate")
    if "data_validity_comment" in work:
        reject(work["data_validity_comment"].notna(), "data_validity_flag")
    accepted_assay_types = set(cfg.get("accepted_assay_types", []))
    if accepted_assay_types and "assay_type" in work:
        reject(~work["assay_type"].isin(accepted_assay_types), "assay_type_not_accepted")

    clean = work[work["exclusion_reason"].eq("")].copy()
    values = pd.to_numeric(clean["standard_value"], errors="coerce")
    clean["standard_value"] = values
    invalid_numeric = ~np.isfinite(values) | (values <= 0)
    clean.loc[invalid_numeric, "exclusion_reason"] = "invalid_standard_value"

    supported = clean["standard_units"].isin(UNIT_TO_MOLAR)
    clean.loc[~supported, "exclusion_reason"] = "unsupported_unit"

    newly_rejected = clean[~clean["exclusion_reason"].eq("")].copy()
    clean = clean[clean["exclusion_reason"].eq("")].copy()
    clean["pIC50"] = [
        ic50_to_pic50(v, u) for v, u in zip(clean["standard_value"], clean["standard_units"], strict=True)
    ]
    if "pchembl_value" in clean:
        clean["pchembl_value_numeric"] = pd.to_numeric(clean["pchembl_value"], errors="coerce")
        clean["pic50_minus_pchembl"] = clean["pIC50"] - clean["pchembl_value_numeric"]

    excluded = pd.concat(
        [work[~work["exclusion_reason"].eq("")], newly_rejected], ignore_index=True
    ).drop_duplicates(subset=[c for c in ["activity_id", "exclusion_reason"] if c in work.columns])
    return clean.reset_index(drop=True), excluded.reset_index(drop=True)


def save_target_resolution(rows: list[dict[str, Any]], output: str | Path) -> Path:
    columns = ["target_symbol", "target_chembl_id", "pref_name", "organism", "target_type", "score_note"]
    return write_dataframe(pd.DataFrame(rows).reindex(columns=columns), output)
