from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from bcvs.http import HTTPConfig, ResilientSession
from bcvs.utils import write_dataframe

LOGGER = logging.getLogger(__name__)


class ZINCClient:
    """Small exact-ID lookups; bulk screening should use downloaded SMI/SDF/CSV shards."""

    def __init__(self, timeout_seconds: float = 60, max_retries: int = 5):
        self.http = ResilientSession(
            HTTPConfig(timeout_seconds=timeout_seconds, max_retries=max_retries, requests_per_second=1.0)
        )

    def fetch_ids(self, zinc_ids: Iterable[str], output_path: str | Path | None = None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for zinc_id in zinc_ids:
            zid = zinc_id.strip().upper()
            if not zid:
                continue
            urls = [
                f"https://zinc20.docking.org/substances/{zid}.json",
                f"https://zinc.docking.org/substances/{zid}.json",
            ]
            payload = None
            for url in urls:
                try:
                    payload = self.http.get_json(url)
                    break
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("ZINC lookup failed at %s: %s", url, exc)
            if payload is None:
                rows.append({"source": "ZINC", "source_id": zid, "lookup_status": "failed"})
                continue
            smiles = (
                payload.get("smiles")
                or payload.get("SMILES")
                or payload.get("canonical_smiles")
                or payload.get("substance", {}).get("smiles")
            )
            rows.append(
                {
                    "source": "ZINC",
                    "source_id": zid,
                    "smiles": smiles,
                    "lookup_status": "ok" if smiles else "no_smiles",
                    "raw_name": payload.get("name") or payload.get("preferred_name"),
                }
            )
        df = pd.DataFrame(rows)
        if output_path is not None:
            write_dataframe(df, output_path)
        return df
