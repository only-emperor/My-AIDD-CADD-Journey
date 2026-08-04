from __future__ import annotations

from bcvs.chem.filters import StructuralAlertFilter, apply_physchem_filter
from bcvs.chem.standardize import MoleculeStandardizer


def test_standardization_largest_fragment() -> None:
    cfg = {
        "cleanup": True,
        "largest_fragment": True,
        "uncharge": True,
        "canonical_tautomer": False,
        "isomeric_smiles": True,
        "allowed_elements": ["H", "C", "N", "O", "Cl"],
    }
    result = MoleculeStandardizer(cfg).standardize("CC[NH+](C)C.[Cl-]")
    assert result.valid
    assert "." not in str(result.standardized_smiles)
    assert result.inchikey


def test_physchem_and_alert_annotations() -> None:
    cfg = {
        "molecular_weight": [50, 600],
        "logp": [-2, 6],
        "tpsa": [0, 200],
        "hbd_max": 6,
        "hba_max": 12,
        "rotatable_bonds_max": 15,
        "formal_charge_abs_max": 2,
        "lipinski_violations_max": 1,
        "synthetic_accessibility_max": 8.0,
    }
    result = apply_physchem_filter("CCOc1ccccc1", cfg)
    assert result.pass_filter
    alerts = StructuralAlertFilter().annotate("CCOc1ccccc1")
    assert set(alerts) >= {"pains_count", "brenk_count", "nih_count"}
