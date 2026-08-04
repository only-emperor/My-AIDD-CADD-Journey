from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd

from bcvs.http import HTTPConfig, ResilientSession
from bcvs.utils import chunks, write_dataframe

LOGGER = logging.getLogger(__name__)

PROPERTY_FIELDS = [
    "CanonicalSMILES",
    "IsomericSMILES",
    "InChIKey",
    "MolecularFormula",
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
]


class PubChemClient:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.base_url = cfg["base_url"].rstrip("/")
        self.http = ResilientSession(
            HTTPConfig(
                timeout_seconds=cfg.get("timeout_seconds", 60),
                max_retries=cfg.get("max_retries", 6),
                requests_per_second=cfg.get("requests_per_second", 4.5),
            )
        )

    def cid_for_name(self, name: str) -> list[int]:
        url = f"{self.base_url}/compound/name/{quote(name, safe='')}/cids/JSON"
        payload = self.http.get_json(url)
        return [int(x) for x in payload.get("IdentifierList", {}).get("CID", [])]

    def properties_for_cids(self, cids: Iterable[int]) -> pd.DataFrame:
        ids = [str(int(cid)) for cid in cids]
        if not ids:
            return pd.DataFrame()
        url = (
            f"{self.base_url}/compound/cid/{','.join(ids)}/property/"
            f"{','.join(PROPERTY_FIELDS)}/JSON"
        )
        payload = self.http.get_json(url)
        rows = payload.get("PropertyTable", {}).get("Properties", [])
        df = pd.DataFrame(rows)
        if not df.empty:
            df["source"] = "PubChem"
            df["source_id"] = df["CID"].map(lambda x: f"CID{x}")
            smiles_col = "ConnectivitySMILES" if "ConnectivitySMILES" in df else "CanonicalSMILES"
            df["smiles"] = df.get("SMILES", df.get("IsomericSMILES", df.get(smiles_col)))
        return df

    def fetch_names(self, names: Iterable[str], output_path: str | Path | None = None) -> pd.DataFrame:
        records: list[pd.DataFrame] = []
        for name in names:
            try:
                cids = self.cid_for_name(name)
                frame = self.properties_for_cids(cids[:10])
                if not frame.empty:
                    frame["query_name"] = name
                    records.append(frame)
            except Exception as exc:  # noqa: BLE001 - continue and preserve audit
                LOGGER.exception("PubChem name lookup failed for %s: %s", name, exc)
        out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        if output_path is not None:
            write_dataframe(out, output_path)
        return out

    def fetch_cids(self, cids: list[int], output_path: str | Path | None = None) -> pd.DataFrame:
        frames = [self.properties_for_cids(batch) for batch in chunks(cids, self.cfg.get("batch_size", 100))]
        out = pd.concat([x for x in frames if not x.empty], ignore_index=True) if frames else pd.DataFrame()
        if output_path is not None:
            write_dataframe(out, output_path)
        return out
