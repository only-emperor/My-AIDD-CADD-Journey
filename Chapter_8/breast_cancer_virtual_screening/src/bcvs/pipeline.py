from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from bcvs.config import ensure_project_layout
from bcvs.data.prepare import prepare_all_targets
from bcvs.models.baseline import train_extra_trees_baseline
from bcvs.models.train import train_ensemble_for_target
from bcvs.plots import generate_all_figures
from bcvs.screen import screen_library
from bcvs.sources.chembl import ChEMBLClient, clean_chembl_activities, save_target_resolution
from bcvs.state import PipelineState
from bcvs.utils import stable_hash, write_dataframe

LOGGER = logging.getLogger(__name__)


def collect_chembl(cfg: dict[str, Any], force: bool = False) -> dict[str, str]:
    paths = ensure_project_layout(cfg)
    state = PipelineState(paths["state"] / "pipeline.sqlite")
    client = ChEMBLClient(cfg["chembl"], paths["raw"])
    outputs: dict[str, str] = {}
    target_rows = []

    for target in cfg["targets"]:
        symbol = str(target["symbol"])
        stage = f"collect_chembl:{symbol}"
        input_hash = stable_hash({"target": target, "chembl": cfg["chembl"]})
        clean_path = paths["interim"] / f"chembl_clean_{symbol}.csv.gz"
        excluded_path = paths["interim"] / f"chembl_excluded_{symbol}.csv.gz"
        if state.stage_done(stage, input_hash) and clean_path.exists() and not force:
            outputs[symbol] = str(clean_path)
            frame = pd.read_csv(clean_path, nrows=1)
            target_rows.append(
                {
                    "target_symbol": symbol,
                    "target_chembl_id": frame["target_chembl_id"].iloc[0] if not frame.empty else None,
                    "pref_name": target.get("name"),
                    "organism": cfg["chembl"]["organism"],
                    "target_type": cfg["chembl"]["target_type"],
                    "score_note": "reused checkpoint",
                }
            )
            continue
        state.start_stage(stage, input_hash, target)
        try:
            resolved = client.resolve_target(
                symbol,
                pinned_id=target.get("target_chembl_id"),
                organism=cfg["chembl"]["organism"],
                target_type=cfg["chembl"]["target_type"],
            )
            chembl_id = str(resolved["target_chembl_id"])
            raw = client.fetch_ic50_activities(chembl_id, symbol)
            clean, excluded = clean_chembl_activities(raw, cfg["chembl"])
            write_dataframe(clean, clean_path)
            write_dataframe(excluded, excluded_path)
            target_rows.append(
                {
                    "target_symbol": symbol,
                    "target_chembl_id": chembl_id,
                    "pref_name": resolved.get("pref_name"),
                    "organism": resolved.get("organism"),
                    "target_type": resolved.get("target_type"),
                    "score_note": "Review this mapping before publication; pin the ID in config.",
                }
            )
            metadata = {"raw_rows": len(raw), "clean_rows": len(clean), "excluded_rows": len(excluded)}
            state.finish_stage(stage, str(clean_path), metadata)
            outputs[symbol] = str(clean_path)
            LOGGER.info("Collected %s (%s): %d clean IC50 rows", symbol, chembl_id, len(clean))
        except Exception as exc:
            state.fail_stage(stage, str(exc))
            raise

    save_target_resolution(target_rows, paths["interim"] / "targets_resolved.csv")
    return outputs


def prepare_training_data(cfg: dict[str, Any], force: bool = False) -> dict[str, str]:
    paths = ensure_project_layout(cfg)
    state = PipelineState(paths["state"] / "pipeline.sqlite")
    stage = "prepare_training"
    clean_paths = sorted(paths["interim"].glob("chembl_clean_*.csv.gz"))
    input_hash = stable_hash(
        {
            "inputs": [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in clean_paths],
            "standardization": cfg["standardization"],
            "chembl": cfg["chembl"],
        }
    )
    if state.stage_done(stage, input_hash) and not force:
        return {p.stem.replace("training_", "").replace(".csv", ""): str(p) for p in paths["processed"].glob("training_*.csv.gz")}
    state.start_stage(stage, input_hash)
    try:
        outputs = prepare_all_targets(
            clean_paths,
            cfg["standardization"],
            cfg["chembl"],
            paths["processed"],
        )
        state.finish_stage(stage, str(paths["processed"]), {k: str(v) for k, v in outputs.items()})
        return {k: str(v) for k, v in outputs.items()}
    except Exception as exc:
        state.fail_stage(stage, str(exc))
        raise


def train_models(cfg: dict[str, Any], force: bool = False, include_baseline: bool = True) -> dict[str, Any]:
    paths = ensure_project_layout(cfg)
    state = PipelineState(paths["state"] / "pipeline.sqlite")
    results: dict[str, Any] = {}
    min_samples = int(cfg["chembl"].get("min_target_samples", 80))
    for training_path in sorted(paths["processed"].glob("training_*.csv.gz")):
        symbol = training_path.name.replace("training_", "").replace(".csv.gz", "")
        n_samples = len(pd.read_csv(training_path, usecols=["pIC50"]))
        if n_samples < min_samples:
            LOGGER.warning("Skipping %s: %d samples < minimum %d", symbol, n_samples, min_samples)
            continue
        stage = f"train:{symbol}"
        input_hash = stable_hash(
            {
                "training_file": (training_path.stat().st_mtime_ns, training_path.stat().st_size),
                "training_config": cfg["training"],
            }
        )
        manifest_path = paths["models"] / symbol / "ensemble_manifest.json"
        if state.stage_done(stage, input_hash) and manifest_path.exists() and not force:
            import json

            results[symbol] = json.loads(manifest_path.read_text(encoding="utf-8"))
            continue
        state.start_stage(stage, input_hash, {"n_samples": n_samples})
        try:
            manifest = train_ensemble_for_target(training_path, symbol, cfg["training"], paths["models"])
            if include_baseline:
                train_extra_trees_baseline(
                    training_path,
                    paths["models"] / symbol / "baseline",
                    cfg["training"],
                    seed=int(cfg["project"].get("seed", 2026)),
                )
            state.finish_stage(stage, str(manifest_path), manifest)
            results[symbol] = manifest
        except Exception as exc:
            state.fail_stage(stage, str(exc))
            raise
    summary_rows = []
    import json

    for manifest_path in sorted(paths["models"].glob("*/ensemble_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for member in manifest.get("members", []):
            test_metrics = member.get("metrics", {}).get("test", {})
            summary_rows.append(
                {
                    "target": manifest.get("target_symbol"),
                    "n_samples": manifest.get("n_samples"),
                    "seed": member.get("seed"),
                    "ensemble_weight": member.get("weight"),
                    "best_epoch": member.get("best_epoch"),
                    "validation_rmse": member.get("validation_rmse"),
                    "test_rmse": test_metrics.get("rmse"),
                    "test_mae": test_metrics.get("mae"),
                    "test_r2": test_metrics.get("r2"),
                    "test_n": test_metrics.get("n"),
                }
            )
    if summary_rows:
        write_dataframe(pd.DataFrame(summary_rows), paths["models"] / "model_summary.csv")
    return results


def run_all(
    cfg: dict[str, Any],
    candidate_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    paths = ensure_project_layout(cfg)
    result: dict[str, Any] = {}
    result["chembl"] = collect_chembl(cfg, force=force)
    result["training_data"] = prepare_training_data(cfg, force=force)
    result["models"] = train_models(cfg, force=force)
    if candidate_path is not None:
        result["screening"] = screen_library(candidate_path, cfg, paths, force=force)
    result["figures"] = generate_all_figures(cfg, paths)
    return result
