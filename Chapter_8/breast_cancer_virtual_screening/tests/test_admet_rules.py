from __future__ import annotations

from bcvs.chem.filters import evaluate_admet_rules


def test_admet_rule_alias_prefix_matching() -> None:
    row = {"hERG_Karim": 0.2, "AMES": 0.1, "HIA_Hou": 0.8}
    rules = {
        "hERG": {"operator": "<=", "threshold": 0.5, "hard": True},
        "Ames": {"operator": "<=", "threshold": 0.5, "hard": True},
        "HIA_Hou": {"operator": ">=", "threshold": 0.5, "hard": False},
    }
    passed, reasons, score, matched = evaluate_admet_rules(row, rules)
    assert passed
    assert not reasons
    assert matched == 3
    assert 0 <= score <= 1
