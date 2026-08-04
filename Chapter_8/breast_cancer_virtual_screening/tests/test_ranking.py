from __future__ import annotations

import pandas as pd

from bcvs.ranking import pareto_ranks, select_diverse_top_hits


def test_pareto_ranking_and_diversity() -> None:
    frame = pd.DataFrame(
        {
            "standardized_smiles": ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "CCO"],
            "inchikey": ["A", "B", "C", "D"],
            "final_score": [0.9, 0.85, 0.8, 0.7],
            "activity_score": [0.9, 0.8, 0.7, 0.6],
            "admet_score": [0.5, 0.6, 0.8, 0.9],
            "qed": [0.7, 0.7, 0.7, 0.7],
            "novelty_score": [0.3, 0.4, 0.7, 0.9],
        }
    )
    ranks = pareto_ranks(frame, ["activity_score", "admet_score", "novelty_score"])
    assert len(ranks) == len(frame)
    selected = select_diverse_top_hits(frame, top_n=3, similarity_cutoff=0.65)
    assert len(selected) == 3
    assert "diversity_cluster" in selected
    assert "pareto_rank" in selected
