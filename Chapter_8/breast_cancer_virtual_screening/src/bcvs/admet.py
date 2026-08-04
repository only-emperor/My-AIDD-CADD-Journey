from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class ADMETPredictor:
    """Lazy ADMET-AI adapter. Install with `pip install -e .[admet]`."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.model = None
        self.error: str | None = None
        self.warned_missing_endpoints: set[str] = set()
        if not enabled:
            return
        try:
            from admet_ai import ADMETModel

            self.model = ADMETModel()
        except Exception as exc:  # noqa: BLE001 - optional dependency/model download
            self.error = str(exc)
            LOGGER.warning("ADMET-AI is unavailable; ADMET endpoints will be missing: %s", exc)

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, smiles: Iterable[str], batch_size: int = 512) -> pd.DataFrame:
        smiles_list = list(smiles)
        if not smiles_list:
            return pd.DataFrame(index=pd.Index([], name="smiles"))
        if self.model is None:
            return pd.DataFrame(index=pd.Index(smiles_list, name="smiles"))
        frames = []
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start : start + batch_size]
            predictions = self.model.predict(smiles=batch)
            if isinstance(predictions, dict):
                predictions = pd.DataFrame([predictions], index=batch)
            elif not isinstance(predictions, pd.DataFrame):
                predictions = pd.DataFrame(predictions)
            predictions = predictions.reset_index(drop=True)
            if len(predictions) != len(batch):
                raise RuntimeError(
                    f"ADMET-AI returned {len(predictions)} rows for {len(batch)} molecules"
                )
            predictions.index = pd.Index(batch, name="smiles")
            frames.append(predictions)
        return pd.concat(frames, axis=0) if frames else pd.DataFrame()


def bbb_rule_score(mw: float, logp: float, tpsa: float, hbd: int) -> float:
    """Transparent non-clinical heuristic used only when a learned BBB endpoint is unavailable."""
    mw_score = 1.0 / (1.0 + np.exp((mw - 450.0) / 35.0))
    tpsa_score = 1.0 / (1.0 + np.exp((tpsa - 90.0) / 10.0))
    logp_low = 1.0 / (1.0 + np.exp(-(logp - 1.0) / 0.4))
    logp_high = 1.0 / (1.0 + np.exp((logp - 4.5) / 0.5))
    hbd_score = 1.0 / (1.0 + np.exp((hbd - 2.0) / 0.5))
    return float(np.mean([mw_score, tpsa_score, logp_low * logp_high, hbd_score]))
