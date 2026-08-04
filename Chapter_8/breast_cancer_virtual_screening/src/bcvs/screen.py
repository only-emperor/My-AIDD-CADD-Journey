from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from bcvs.admet import ADMETPredictor, bbb_rule_score
from bcvs.chem.filters import StructuralAlertFilter, apply_physchem_filter, evaluate_admet_rules
from bcvs.chem.standardize import MoleculeStandardizer
from bcvs.data.io import infer_smiles_column, iter_candidate_chunks
from bcvs.models.ensemble import GNNEnsemble
from bcvs.ranking import select_diverse_top_hits
from bcvs.state import PipelineState
from bcvs.utils import atomic_write_json, sha256_file, stable_hash, write_dataframe

LOGGER = logging.getLogger(__name__)


class ScreeningDeduper:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS seen_molecules(
                    inchikey TEXT PRIMARY KEY,
                    first_chunk INTEGER NOT NULL,
                    first_source_id TEXT
                )"""
            )

    def previously_seen(self, inchikeys: list[str], chunk_id: int) -> set[str]:
        keys = [x for x in inchikeys if x]
        if not keys:
            return set()
        found: set[str] = set()
        with sqlite3.connect(self.db_path) as con:
            for start in range(0, len(keys), 800):
                batch = keys[start : start + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = con.execute(
                    f"SELECT inchikey, first_chunk FROM seen_molecules WHERE inchikey IN ({placeholders})",
                    batch,
                ).fetchall()
                found.update(key for key, first_chunk in rows if int(first_chunk) != chunk_id)
        return found

    def register(self, rows: list[tuple[str, int, str]]) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executemany(
                "INSERT OR IGNORE INTO seen_molecules(inchikey,first_chunk,first_source_id) VALUES(?,?,?)",
                rows,
            )


class ApplicabilityDomain:
    def __init__(self, training_smiles: list[str], limit: int = 2500, seed: int = 2026):
        if len(training_smiles) > limit:
            rng = np.random.default_rng(seed)
            selected = rng.choice(len(training_smiles), size=limit, replace=False)
            training_smiles = [training_smiles[int(i)] for i in selected]
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.reference_fps = []
        for smi in training_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                self.reference_fps.append(self.generator.GetFingerprint(mol))

    def max_similarity(self, smiles: list[str]) -> np.ndarray:
        values = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None or not self.reference_fps:
                values.append(float("nan"))
                continue
            fp = self.generator.GetFingerprint(mol)
            values.append(max(DataStructs.BulkTanimotoSimilarity(fp, self.reference_fps)))
        return np.asarray(values, dtype=float)


def _sigmoid_activity(pic50: np.ndarray, threshold: float, scale: float = 0.75) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(pic50 - threshold) / scale))


def _resolve_bbb_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(c).lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _load_models_and_domains(
    model_root: Path, target_cfgs: list[dict[str, Any]], screening_cfg: dict[str, Any]
) -> tuple[dict[str, GNNEnsemble], dict[str, ApplicabilityDomain], dict[str, float]]:
    ensembles: dict[str, GNNEnsemble] = {}
    domains: dict[str, ApplicabilityDomain] = {}
    target_weights = {str(t["symbol"]): float(t.get("weight", 1.0)) for t in target_cfgs}
    for target in target_cfgs:
        symbol = str(target["symbol"])
        manifest_path = model_root / symbol / "ensemble_manifest.json"
        if not manifest_path.exists():
            LOGGER.warning("No ensemble manifest for %s; target will be skipped", symbol)
            continue
        ensemble = GNNEnsemble(manifest_path)
        ensembles[symbol] = ensemble
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training_path = Path(manifest["training_table"])
        if training_path.exists():
            train = pd.read_csv(training_path, usecols=["standardized_smiles"])
            domains[symbol] = ApplicabilityDomain(
                train["standardized_smiles"].astype(str).tolist(),
                limit=int(screening_cfg.get("applicability_reference_limit", 2500)),
            )
    if not ensembles:
        raise RuntimeError(f"No trained ensembles found under {model_root}")
    return ensembles, domains, target_weights


def _standardize_and_filter_chunk(
    chunk: pd.DataFrame,
    chunk_id: int,
    smiles_column: str,
    standardizer: MoleculeStandardizer,
    alerts: StructuralAlertFilter,
    cfg: dict[str, Any],
    deduper: ScreeningDeduper,
) -> pd.DataFrame:
    frame = chunk.reset_index(drop=True).copy()
    input_smiles = frame[smiles_column].copy()
    reserved = {
        "standardized_smiles", "inchikey", "valid", "standardization_error",
        "atom_count", "heavy_atom_count", "input_smiles"
    }
    rename_map = {column: f"source_{column}" for column in reserved if column in frame.columns}
    frame = frame.rename(columns=rename_map)
    frame["input_smiles"] = input_smiles
    frame["source"] = frame.get("source", "local")
    if "source_id" not in frame:
        frame["source_id"] = [f"chunk{chunk_id}_row{i}" for i in range(len(frame))]
    else:
        generated = pd.Series([f"chunk{chunk_id}_row{i}" for i in range(len(frame))], index=frame.index)
        frame["source_id"] = frame["source_id"].fillna(generated).astype(str)
    frame["screen_chunk"] = chunk_id
    std_rows = [standardizer.standardize(x).to_dict() for x in frame["input_smiles"]]
    std_frame = pd.DataFrame(std_rows).drop(columns=["input_smiles"], errors="ignore")
    frame = pd.concat([frame, std_frame], axis=1)
    valid_keys = frame.loc[frame["valid"], "inchikey"].fillna("").astype(str).tolist()
    previous = deduper.previously_seen(valid_keys, chunk_id)
    frame["duplicate"] = False
    local_seen: set[str] = set()
    for idx in frame.index[frame["valid"]]:
        key = str(frame.at[idx, "inchikey"] or "")
        if key in previous or key in local_seen:
            frame.at[idx, "duplicate"] = True
        else:
            local_seen.add(key)

    for col in [
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "ring_count",
        "aromatic_ring_count",
        "fraction_csp3",
        "formal_charge",
        "qed",
        "sa_score",
        "sa_desirability",
        "lipinski_violations",
        "pains_count",
        "pains_alerts",
        "brenk_count",
        "brenk_alerts",
        "nih_count",
        "nih_alerts",
    ]:
        frame[col] = np.nan if not col.endswith("alerts") else ""
    frame["physchem_pass"] = False
    frame["structural_alert_pass"] = False
    frame["cheap_filter_reasons"] = ""

    for idx in frame.index[frame["valid"] & ~frame["duplicate"]]:
        smi = str(frame.at[idx, "standardized_smiles"])
        phys = apply_physchem_filter(smi, cfg["physchem"])
        alert_values = alerts.annotate(smi)
        for key, value in {**phys.annotations, **alert_values}.items():
            frame.at[idx, key] = value
        alert_reasons = []
        structural_cfg = cfg["structural_alerts"]
        for catalog in ["pains", "brenk", "nih"]:
            if structural_cfg.get(f"{catalog}_hard_filter", False) and int(alert_values[f"{catalog}_count"]) > 0:
                alert_reasons.append(catalog)
        frame.at[idx, "physchem_pass"] = phys.pass_filter
        frame.at[idx, "structural_alert_pass"] = not alert_reasons
        frame.at[idx, "cheap_filter_reasons"] = ";".join(phys.reasons + alert_reasons)
    return frame


def _predict_models(
    frame: pd.DataFrame,
    ensembles: dict[str, GNNEnsemble],
    domains: dict[str, ApplicabilityDomain],
    target_weights: dict[str, float],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    eligible = frame["valid"] & ~frame["duplicate"] & frame["physchem_pass"] & frame["structural_alert_pass"]
    eligible_idx = frame.index[eligible]
    smiles = frame.loc[eligible_idx, "standardized_smiles"].astype(str).tolist()
    threshold = float(cfg["screening"]["activity_threshold_pic50"])
    uncertainty_max = float(cfg["screening"]["maximum_uncertainty"])
    target_activity_scores = []
    target_similarities = []
    target_passes = []

    for symbol, ensemble in ensembles.items():
        mean_col = f"pred_pic50_{symbol}"
        std_col = f"pred_uncertainty_{symbol}"
        sim_col = f"max_tanimoto_{symbol}"
        score_col = f"activity_score_{symbol}"
        frame[mean_col] = np.nan
        frame[std_col] = np.nan
        frame[sim_col] = np.nan
        frame[score_col] = np.nan
        if not smiles:
            continue
        pred = ensemble.predict(smiles, batch_size=int(cfg["training"]["batch_size"]) * 2)
        similarities = (
            domains[symbol].max_similarity(smiles)
            if symbol in domains
            else np.full(len(smiles), np.nan, dtype=float)
        )
        score = _sigmoid_activity(pred["mean"], threshold) * np.exp(-pred["std"])
        score *= float(target_weights.get(symbol, 1.0))
        frame.loc[eligible_idx, mean_col] = pred["mean"]
        frame.loc[eligible_idx, std_col] = pred["std"]
        frame.loc[eligible_idx, sim_col] = similarities
        frame.loc[eligible_idx, score_col] = score
        target_activity_scores.append(score)
        target_similarities.append(similarities)
        target_passes.append((pred["mean"] >= threshold) & (pred["std"] <= uncertainty_max))

    frame["activity_score"] = np.nan
    frame["max_training_tanimoto"] = np.nan
    frame["gnn_pass"] = False
    if smiles and target_activity_scores:
        score_matrix = np.vstack(target_activity_scores)
        sim_matrix = np.vstack(target_similarities)
        pass_matrix = np.vstack(target_passes)
        if cfg["screening"].get("target_aggregation", "weighted_max") == "weighted_mean":
            overall = np.nanmean(score_matrix, axis=0)
        else:
            overall = np.nanmax(score_matrix, axis=0)
        frame.loc[eligible_idx, "activity_score"] = overall
        frame.loc[eligible_idx, "max_training_tanimoto"] = np.nanmax(sim_matrix, axis=0)
        frame.loc[eligible_idx, "gnn_pass"] = np.any(pass_matrix, axis=0)
    return frame


def _predict_admet_and_score(
    frame: pd.DataFrame,
    admet: ADMETPredictor,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    eligible = frame["valid"] & ~frame["duplicate"] & frame["physchem_pass"] & frame["structural_alert_pass"]
    eligible_idx = frame.index[eligible]
    smiles = frame.loc[eligible_idx, "standardized_smiles"].astype(str).tolist()
    predictions = admet.predict(smiles, batch_size=int(cfg["admet"].get("batch_size", 512)))
    if not predictions.empty:
        for col in predictions.columns:
            frame[col] = np.nan
            frame.loc[eligible_idx, col] = predictions[col].to_numpy()

    frame["admet_pass"] = True
    frame["admet_score"] = np.nan
    frame["admet_rule_count"] = 0
    frame["admet_reasons"] = ""
    rules = cfg["admet"].get("rules", {})
    available_lower = {str(column).lower() for column in frame.columns}
    missing_rules = [
        endpoint for endpoint in rules
        if endpoint.lower() not in available_lower
        and not any(column.startswith(endpoint.lower() + "_") for column in available_lower)
    ]
    unseen_missing = [name for name in missing_rules if name not in admet.warned_missing_endpoints]
    if unseen_missing and admet.available and cfg["admet"].get("missing_endpoint_policy", "warn") == "warn":
        LOGGER.warning("ADMET endpoints not available in this run: %s", ", ".join(unseen_missing))
        admet.warned_missing_endpoints.update(unseen_missing)
    for idx in eligible_idx:
        row = frame.loc[idx].to_dict()
        hard_pass, reasons, score, matched = evaluate_admet_rules(row, rules)
        frame.at[idx, "admet_pass"] = hard_pass
        frame.at[idx, "admet_score"] = score
        frame.at[idx, "admet_rule_count"] = matched
        if missing_rules and cfg["admet"].get("missing_endpoint_policy") == "fail":
            hard_pass = False
            reasons.extend(f"admet_missing:{name}" for name in missing_rules)
            frame.at[idx, "admet_pass"] = False
        frame.at[idx, "admet_reasons"] = ";".join(reasons)

    frame["bbb_rule_score"] = np.nan
    for idx in eligible_idx:
        frame.at[idx, "bbb_rule_score"] = bbb_rule_score(
            float(frame.at[idx, "mw"]),
            float(frame.at[idx, "logp"]),
            float(frame.at[idx, "tpsa"]),
            int(frame.at[idx, "hbd"]),
        )
    bbb_cfg = cfg["bbb"]
    endpoint = _resolve_bbb_column(frame, list(bbb_cfg.get("endpoint_candidates", [])))
    frame["bbb_endpoint_used"] = endpoint or "BBB_rule_score"
    bbb_values = pd.to_numeric(frame[endpoint], errors="coerce") if endpoint else frame["bbb_rule_score"]
    frame["bbb_score"] = bbb_values
    if bbb_cfg.get("desired", "permeable") == "permeable":
        frame["bbb_pass"] = bbb_values >= float(bbb_cfg.get("threshold", 0.5))
    else:
        frame["bbb_pass"] = bbb_values < float(bbb_cfg.get("threshold", 0.5))
    frame.loc[~eligible, "bbb_pass"] = False
    return frame


def _finalize_scores(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    ad_score = pd.to_numeric(frame["admet_score"], errors="coerce").fillna(0.5).clip(0, 1)
    similarity = pd.to_numeric(frame["max_training_tanimoto"], errors="coerce").fillna(0.0).clip(0, 1)
    activity = pd.to_numeric(frame["activity_score"], errors="coerce").fillna(0.0).clip(0, 1)
    qed = pd.to_numeric(frame["qed"], errors="coerce").fillna(0.0).clip(0, 1)
    sa_desirability = pd.to_numeric(frame.get("sa_desirability", 0.5), errors="coerce").fillna(0.5).clip(0, 1)
    druglike = 0.65 * qed + 0.35 * sa_desirability
    frame["druglikeness_score"] = druglike
    novelty = 1.0 - similarity
    domain_min = float(cfg["screening"]["applicability_domain_min_tanimoto"])
    frame["applicability_pass"] = similarity >= domain_min
    frame["novelty_score"] = novelty
    weights = cfg["screening"]["final_weights"]
    frame["final_score"] = (
        float(weights["activity"]) * activity
        + float(weights["admet"]) * ad_score
        + float(weights["druglikeness"]) * druglike
        + float(weights["applicability"]) * similarity
        + float(weights["novelty"]) * novelty
    )
    hard_bbb = bool(cfg["bbb"].get("enabled", True) and cfg["bbb"].get("hard_filter", False))
    frame["final_pass"] = (
        frame["valid"]
        & ~frame["duplicate"]
        & frame["physchem_pass"]
        & frame["structural_alert_pass"]
        & frame["gnn_pass"]
        & frame["admet_pass"]
        & frame["applicability_pass"]
        & (frame["bbb_pass"] if hard_bbb else True)
    )

    def rejection(row: pd.Series) -> str:
        reasons = []
        if not bool(row.get("valid", False)):
            reasons.append(f"invalid:{row.get('standardization_error', '')}")
        if bool(row.get("duplicate", False)):
            reasons.append("duplicate")
        if not bool(row.get("physchem_pass", False)):
            reasons.append("physchem")
        if not bool(row.get("structural_alert_pass", False)):
            reasons.append("structural_alert")
        if not bool(row.get("gnn_pass", False)):
            reasons.append("activity_or_uncertainty")
        if not bool(row.get("admet_pass", True)):
            reasons.append("admet")
        if not bool(row.get("applicability_pass", False)):
            reasons.append("outside_applicability_domain")
        if hard_bbb and not bool(row.get("bbb_pass", False)):
            reasons.append("bbb")
        extra = [row.get("cheap_filter_reasons", ""), row.get("admet_reasons", "")]
        reasons.extend(x for x in extra if isinstance(x, str) and x)
        return ";".join(dict.fromkeys(reasons))

    frame["rejection_reason"] = frame.apply(rejection, axis=1)
    return frame


def screen_library(
    candidate_path: str | Path,
    cfg: dict[str, Any],
    paths: dict[str, Path],
    force: bool = False,
) -> dict[str, Any]:
    candidate_path = Path(candidate_path).expanduser().resolve()
    candidate_sha256 = sha256_file(candidate_path)
    safe_name = candidate_path.name.replace(".", "_")
    run_name = f"{safe_name}_{candidate_sha256[:10]}"
    output_dir = paths["screening"] / run_name
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    state = PipelineState(paths["state"] / "pipeline.sqlite")
    stage = f"screen:{run_name}"
    model_fingerprints = {
        str(path.relative_to(paths["models"])): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths["models"].glob("*/ensemble_manifest.json")
    }
    input_hash = stable_hash(
        {
            "candidate_sha256": candidate_sha256,
            "standardization": cfg["standardization"],
            "physchem": cfg["physchem"],
            "structural_alerts": cfg["structural_alerts"],
            "screening": cfg["screening"],
            "admet": cfg["admet"],
            "bbb": cfg["bbb"],
            "target_weights": cfg["targets"],
            "model_manifests": model_fingerprints,
        }
    )
    if state.stage_done(stage, input_hash) and not force:
        manifest_path = output_dir / "screening_manifest.json"
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    state.start_stage(stage, input_hash, {"candidate": str(candidate_path)})

    standardizer = MoleculeStandardizer(cfg["standardization"])
    alerts = StructuralAlertFilter()
    deduper = ScreeningDeduper(output_dir / "dedupe.sqlite")
    admet = ADMETPredictor(cfg["admet"].get("enabled", True))
    ensembles, domains, target_weights = _load_models_and_domains(
        paths["models"], cfg["targets"], cfg["screening"]
    )
    part_paths: list[str] = []
    funnel_records: list[dict[str, Any]] = []
    register_rows: list[tuple[str, int, str]] = []

    try:
        for chunk_id, chunk in enumerate(
            iter_candidate_chunks(candidate_path, int(cfg["screening"]["chunk_size"]))
        ):
            part_path = parts_dir / f"part_{chunk_id:06d}.csv.gz"
            if state.chunk_done(stage, chunk_id) and part_path.exists() and not force:
                part_paths.append(str(part_path))
                continue
            state.start_chunk(stage, chunk_id, {"rows": len(chunk)})
            try:
                smiles_column = infer_smiles_column(list(chunk.columns))
                frame = _standardize_and_filter_chunk(
                    chunk, chunk_id, smiles_column, standardizer, alerts, cfg, deduper
                )
                frame = _predict_models(frame, ensembles, domains, target_weights, cfg)
                frame = _predict_admet_and_score(frame, admet, cfg)
                frame = _finalize_scores(frame, cfg)
                write_dataframe(frame, part_path)
                unique_rows = frame[frame["valid"] & ~frame["duplicate"]]
                register_rows = [
                    (str(row.inchikey), chunk_id, str(row.source_id))
                    for row in unique_rows[["inchikey", "source_id"]].itertuples(index=False)
                ]
                deduper.register(register_rows)
                valid_mask = frame["valid"]
                unique_mask = valid_mask & ~frame["duplicate"]
                physchem_mask = unique_mask & frame["physchem_pass"]
                alerts_mask = physchem_mask & frame["structural_alert_pass"]
                gnn_mask = alerts_mask & frame["gnn_pass"]
                admet_mask = gnn_mask & frame["admet_pass"]
                bbb_is_hard = bool(cfg["bbb"].get("enabled", True) and cfg["bbb"].get("hard_filter", False))
                bbb_policy_mask = admet_mask & (frame["bbb_pass"] if bbb_is_hard else True)
                funnel_records.append(
                    {
                        "chunk": chunk_id,
                        "input": int(len(frame)),
                        "valid": int(valid_mask.sum()),
                        "unique": int(unique_mask.sum()),
                        "physchem": int(physchem_mask.sum()),
                        "alerts": int(alerts_mask.sum()),
                        "gnn": int(gnn_mask.sum()),
                        "admet": int(admet_mask.sum()),
                        "bbb": int(bbb_policy_mask.sum()),
                        "bbb_permeable": int((admet_mask & frame["bbb_pass"]).sum()),
                        "final": int(frame["final_pass"].sum()),
                    }
                )
                part_paths.append(str(part_path))
                state.finish_chunk(stage, chunk_id, str(part_path), funnel_records[-1])
                LOGGER.info("Screened chunk %d: %d rows, %d final hits", chunk_id, len(frame), int(frame["final_pass"].sum()))
            except Exception as exc:
                state.fail_chunk(stage, chunk_id, str(exc))
                raise

        top_candidates = []
        rejection_counts: dict[str, int] = {}
        top_n = int(cfg["screening"]["top_n"])
        pool_multiplier = int(cfg["screening"].get("diversity_pool_multiplier", 5))
        for part in part_paths:
            frame = pd.read_csv(part)
            passed = frame[frame["final_pass"]].nlargest(top_n * pool_multiplier, "final_score")
            if not passed.empty:
                top_candidates.append(passed)
            for reason, count in frame["rejection_reason"].fillna("").value_counts().items():
                rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + int(count)
        if top_candidates:
            candidate_pool = (
                pd.concat(top_candidates, ignore_index=True)
                .sort_values("final_score", ascending=False)
                .drop_duplicates("inchikey")
                .head(top_n * pool_multiplier)
            )
            top_hits = select_diverse_top_hits(
                candidate_pool,
                top_n=top_n,
                similarity_cutoff=float(cfg["screening"].get("diversity_tanimoto_cutoff", 0.65)),
            )
        else:
            top_hits = pd.DataFrame()
        top_path = write_dataframe(top_hits, output_dir / "top_hits.csv")
        funnel = pd.DataFrame(funnel_records)
        funnel_path = write_dataframe(funnel, output_dir / "screening_funnel_by_chunk.csv")
        manifest = {
            "candidate_path": str(candidate_path),
            "candidate_sha256": candidate_sha256,
            "parts": part_paths,
            "top_hits": str(top_path),
            "funnel": str(funnel_path),
            "rejection_counts": rejection_counts,
            "admet_ai_available": admet.available,
            "admet_ai_error": admet.error,
            "targets_screened": sorted(ensembles),
        }
        manifest_path = output_dir / "screening_manifest.json"
        atomic_write_json(manifest, manifest_path)
        state.finish_stage(stage, str(manifest_path), manifest)
        return manifest
    except Exception as exc:
        state.fail_stage(stage, str(exc))
        raise
