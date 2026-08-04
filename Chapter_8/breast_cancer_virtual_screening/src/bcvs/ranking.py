from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina


def assign_butina_clusters(
    frame: pd.DataFrame,
    smiles_column: str = "standardized_smiles",
    similarity_cutoff: float = 0.65,
) -> pd.DataFrame:
    """Assign ECFP4 Butina clusters; input order should already reflect priority."""
    out = frame.reset_index(drop=True).copy()
    if out.empty:
        out["diversity_cluster"] = pd.Series(dtype="int64")
        out["cluster_size"] = pd.Series(dtype="int64")
        return out
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    valid_indices = []
    for idx, smi in enumerate(out[smiles_column].astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(generator.GetFingerprint(mol))
            valid_indices.append(idx)
    distances = []
    for i in range(1, len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend(1.0 - value for value in similarities)
    distance_cutoff = 1.0 - float(similarity_cutoff)
    clusters = Butina.ClusterData(distances, len(fps), distance_cutoff, isDistData=True)
    out["diversity_cluster"] = -1
    out["cluster_size"] = 1
    for cluster_id, members in enumerate(clusters):
        mapped = [valid_indices[int(member)] for member in members]
        out.loc[mapped, "diversity_cluster"] = cluster_id
        out.loc[mapped, "cluster_size"] = len(mapped)
    return out


def pareto_ranks(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return non-dominated sorting ranks for objectives that are all maximized."""
    if frame.empty:
        return np.array([], dtype=int)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(-np.inf).to_numpy(float)
    n = len(values)
    ranks = np.full(n, -1, dtype=int)
    remaining = np.arange(n)
    rank = 0
    while len(remaining):
        current_values = values[remaining]
        dominated = np.zeros(len(remaining), dtype=bool)
        for i in range(len(remaining)):
            if dominated[i]:
                continue
            # j dominates i if j is no worse in all objectives and better in at least one.
            no_worse = np.all(current_values >= current_values[i], axis=1)
            strictly_better = np.any(current_values > current_values[i], axis=1)
            if np.any(no_worse & strictly_better):
                dominated[i] = True
        front = remaining[~dominated]
        if len(front) == 0:  # numerical guard
            front = remaining[:1]
        ranks[front] = rank
        remaining = remaining[dominated]
        rank += 1
    return ranks


def select_diverse_top_hits(
    frame: pd.DataFrame,
    top_n: int,
    similarity_cutoff: float,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.sort_values("final_score", ascending=False).drop_duplicates("inchikey").reset_index(drop=True)
    ranked = assign_butina_clusters(ranked, similarity_cutoff=similarity_cutoff)
    objectives = [c for c in ["activity_score", "admet_score", "qed", "novelty_score"] if c in ranked]
    ranked["pareto_rank"] = pareto_ranks(ranked, objectives) if objectives else 0
    representatives = (
        ranked.sort_values(["pareto_rank", "final_score"], ascending=[True, False])
        .drop_duplicates("diversity_cluster")
        .head(top_n)
        .copy()
    )
    representatives["cluster_representative"] = True
    if len(representatives) < top_n:
        used = set(representatives.index)
        fill = ranked.loc[~ranked.index.isin(used)].head(top_n - len(representatives)).copy()
        fill["cluster_representative"] = False
        representatives = pd.concat([representatives, fill], ignore_index=True)
    representatives = representatives.sort_values(
        ["pareto_rank", "final_score"], ascending=[True, False]
    ).reset_index(drop=True)
    representatives.insert(0, "final_rank", np.arange(1, len(representatives) + 1))
    return representatives
