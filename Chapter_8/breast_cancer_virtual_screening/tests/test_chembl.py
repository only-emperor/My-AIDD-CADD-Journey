from __future__ import annotations

import math

import pandas as pd

from bcvs.sources.chembl import clean_chembl_activities, ic50_to_pic50


def test_ic50_to_pic50_units() -> None:
    assert math.isclose(ic50_to_pic50(10, "nM"), 8.0)
    assert math.isclose(ic50_to_pic50(1, "uM"), 6.0)
    assert math.isclose(ic50_to_pic50(0.1, "mM"), 4.0)


def test_clean_chembl_activities() -> None:
    df = pd.DataFrame(
        {
            "activity_id": [1, 2, 3],
            "canonical_smiles": ["CCO", "CCN", "CCC"],
            "standard_value": [10, 1, 100],
            "standard_units": ["nM", "uM", "nM"],
            "standard_type": ["IC50", "IC50", "IC50"],
            "standard_relation": ["=", ">", "="],
            "standard_flag": [1, 1, 0],
            "potential_duplicate": [0, 0, 0],
            "data_validity_comment": [None, None, None],
            "assay_type": ["B", "B", "B"],
        }
    )
    cfg = {
        "activity_type": "IC50",
        "only_exact_relation": True,
        "require_standard_flag": True,
        "remove_potential_duplicates": True,
        "accepted_assay_types": ["B"],
    }
    clean, excluded = clean_chembl_activities(df, cfg)
    assert len(clean) == 1
    assert math.isclose(clean.loc[0, "pIC50"], 8.0)
    assert set(excluded["exclusion_reason"]) == {"censored_relation", "nonstandard_record"}
