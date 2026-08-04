from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator
from sklearn.decomposition import PCA

LOGGER = logging.getLogger(__name__)

NATURE_PALETTE = ["#3C5488", "#E64B35", "#00A087", "#4DBBD5", "#F39B7F", "#8491B4", "#91D1C2"]


def configure_publication_style(cfg: dict[str, Any]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [cfg.get("font_family", "Times New Roman"), "Times", "DejaVu Serif"],
            "font.size": cfg.get("base_font_size", 8),
            "axes.labelsize": cfg.get("base_font_size", 8),
            "axes.titlesize": cfg.get("base_font_size", 8),
            "xtick.labelsize": cfg.get("base_font_size", 8) - 1,
            "ytick.labelsize": cfg.get("base_font_size", 8) - 1,
            "legend.fontsize": cfg.get("base_font_size", 8) - 1,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.constrained_layout.use": True,
        }
    )


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, stem: str | Path, cfg: dict[str, Any]) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in cfg.get("format", ["png", "pdf", "svg"]):
        path = stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt == "png":
            kwargs["dpi"] = int(cfg.get("dpi", 600))
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_target_data_volume(training_files: Iterable[Path], output: Path, cfg: dict[str, Any]) -> None:
    rows = []
    for path in training_files:
        df = pd.read_csv(path)
        if not df.empty:
            rows.append({"Target": str(df["target_symbol"].iloc[0]), "Molecules": len(df)})
    table = pd.DataFrame(rows).sort_values("Molecules", ascending=True)
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(cfg["figure_width_in"], max(2.4, 0.34 * len(table))))
    ax.barh(table["Target"], table["Molecules"], color=NATURE_PALETTE[0], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Unique standardized molecules")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _despine(ax)
    for i, value in enumerate(table["Molecules"]):
        ax.text(value, i, f" {value:,}", va="center", ha="left", fontsize=6.5)
    save_figure(fig, output, cfg)


def plot_pic50_distributions(training_files: Iterable[Path], output: Path, cfg: dict[str, Any]) -> None:
    frames = [pd.read_csv(path, usecols=["target_symbol", "pIC50"]) for path in training_files]
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    targets = sorted(data["target_symbol"].unique())
    fig, ax = plt.subplots(figsize=(cfg["wide_figure_width_in"], 3.2))
    bins = np.linspace(max(2, data["pIC50"].min()), min(12, data["pIC50"].max()), 36)
    for idx, target in enumerate(targets):
        values = data.loc[data["target_symbol"] == target, "pIC50"].dropna()
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.2,
            density=True,
            label=f"{target} (n={len(values):,})",
            color=NATURE_PALETTE[idx % len(NATURE_PALETTE)],
        )
    ax.set_xlabel(r"pIC$_{50}$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, ncol=min(4, len(targets)), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    _despine(ax)
    save_figure(fig, output, cfg)


def _fingerprint_array(smiles: list[str], n_bits: int = 1024) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    arr = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr[idx])
    return arr


def plot_chemical_space(training_files: Iterable[Path], output: Path, cfg: dict[str, Any], max_points: int = 6000) -> None:
    frames = []
    for path in training_files:
        df = pd.read_csv(path, usecols=["target_symbol", "standardized_smiles"])
        frames.append(df)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    if len(data) > max_points:
        data = data.sample(max_points, random_state=2026)
    x = _fingerprint_array(data["standardized_smiles"].astype(str).tolist())
    coords = PCA(n_components=2, random_state=2026, svd_solver="randomized").fit_transform(x)
    fig, ax = plt.subplots(figsize=(cfg["figure_width_in"], 3.2))
    for idx, target in enumerate(sorted(data["target_symbol"].unique())):
        mask = data["target_symbol"].to_numpy() == target
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=7,
            alpha=0.55,
            linewidths=0,
            label=target,
            color=NATURE_PALETTE[idx % len(NATURE_PALETTE)],
        )
    ax.set_xlabel("PCA 1 (Morgan fingerprint)")
    ax.set_ylabel("PCA 2 (Morgan fingerprint)")
    ax.legend(frameon=False, markerscale=1.5, ncol=2)
    _despine(ax)
    save_figure(fig, output, cfg)


def plot_model_performance(model_root: Path, output: Path, cfg: dict[str, Any]) -> None:
    frames = []
    for pred_path in model_root.glob("*/predictions_seed_*.csv"):
        target = pred_path.parent.name
        df = pd.read_csv(pred_path)
        df = df[df["split"] == "test"].copy()
        df["target"] = target
        df["seed"] = pred_path.stem.split("_")[-1]
        frames.append(df)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(cfg["figure_width_in"], 3.3))
    for idx, target in enumerate(sorted(data["target"].unique())):
        subset = data[data["target"] == target]
        ax.scatter(
            subset["observed_pic50"],
            subset["predicted_pic50"],
            s=8,
            alpha=0.45,
            linewidths=0,
            label=target,
            color=NATURE_PALETTE[idx % len(NATURE_PALETTE)],
        )
    low = float(min(data["observed_pic50"].min(), data["predicted_pic50"].min()))
    high = float(max(data["observed_pic50"].max(), data["predicted_pic50"].max()))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=0.8, color="black")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"Observed pIC$_{50}$")
    ax.set_ylabel(r"Predicted pIC$_{50}$")
    ax.legend(frameon=False, ncol=2)
    _despine(ax)
    save_figure(fig, output, cfg)


def plot_residuals(model_root: Path, output: Path, cfg: dict[str, Any]) -> None:
    frames = []
    for pred_path in model_root.glob("*/predictions_seed_*.csv"):
        target = pred_path.parent.name
        df = pd.read_csv(pred_path)
        df = df[df["split"] == "test"].copy()
        df["target"] = target
        frames.append(df)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(cfg["figure_width_in"], 3.0))
    for idx, target in enumerate(sorted(data["target"].unique())):
        subset = data[data["target"] == target]
        ax.scatter(
            subset["predicted_pic50"],
            subset["residual"],
            s=8,
            alpha=0.45,
            linewidths=0,
            label=target,
            color=NATURE_PALETTE[idx % len(NATURE_PALETTE)],
        )
    ax.axhline(0, linewidth=0.8, linestyle="--", color="black")
    ax.set_xlabel(r"Predicted pIC$_{50}$")
    ax.set_ylabel("Residual (observed − predicted)")
    _despine(ax)
    save_figure(fig, output, cfg)


def plot_ensemble_weights(model_root: Path, output: Path, cfg: dict[str, Any]) -> None:
    rows = []
    for manifest_path in model_root.glob("*/ensemble_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for member in manifest["members"]:
            rows.append(
                {
                    "target": manifest["target_symbol"],
                    "seed": str(member["seed"]),
                    "weight": float(member["weight"]),
                    "validation_rmse": float(member["validation_rmse"]),
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return
    targets = sorted(data["target"].unique())
    seeds = sorted(data["seed"].unique())
    x = np.arange(len(targets))
    width = 0.75 / max(1, len(seeds))
    fig, ax = plt.subplots(figsize=(cfg["wide_figure_width_in"], 3.0))
    for idx, seed in enumerate(seeds):
        vals = [
            data.loc[(data["target"] == target) & (data["seed"] == seed), "weight"].iloc[0]
            if ((data["target"] == target) & (data["seed"] == seed)).any()
            else 0
            for target in targets
        ]
        ax.bar(
            x + (idx - (len(seeds) - 1) / 2) * width,
            vals,
            width,
            label=f"Seed {seed}",
            color=NATURE_PALETTE[idx % len(NATURE_PALETTE)],
            edgecolor="black",
            linewidth=0.35,
        )
    ax.set_xticks(x, targets, rotation=30, ha="right")
    ax.set_ylabel("Ensemble weight")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=len(seeds))
    _despine(ax)
    save_figure(fig, output, cfg)


def plot_screening_funnel(funnel_path: Path, output: Path, cfg: dict[str, Any]) -> None:
    data = pd.read_csv(funnel_path)
    if data.empty:
        return
    stages = ["input", "valid", "unique", "physchem", "alerts", "gnn", "admet", "bbb", "final"]
    counts = data[stages].sum()
    labels = ["Input", "Valid", "Unique", "Physchem", "Alerts", "GNN", "ADMET", "BBB policy", "Final"]
    fig, ax = plt.subplots(figsize=(cfg["wide_figure_width_in"], 3.0))
    bars = ax.bar(labels, counts, color=NATURE_PALETTE[0], edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Molecules")
    ax.set_yscale("log" if counts.max() / max(counts[counts > 0].min(), 1) > 100 else "linear")
    ax.tick_params(axis="x", rotation=35)
    _despine(ax)
    for bar, value in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(value):,}", ha="center", va="bottom", fontsize=6.5)
    save_figure(fig, output, cfg)


def plot_top_hit_heatmap(top_hits_path: Path, output: Path, cfg: dict[str, Any], top_n: int = 30) -> None:
    data = pd.read_csv(top_hits_path).head(top_n)
    pred_cols = [c for c in data.columns if c.startswith("pred_pic50_")]
    if data.empty or not pred_cols:
        return
    matrix = data[pred_cols].to_numpy(dtype=float)
    labels = [c.replace("pred_pic50_", "") for c in pred_cols]
    row_labels = data.get("source_id", pd.Series([f"Hit {i+1}" for i in range(len(data))])).astype(str).tolist()
    fig_height = max(3.0, 0.18 * len(data) + 1.0)
    fig, ax = plt.subplots(figsize=(cfg["figure_width_in"], fig_height))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Breast-cancer target")
    ax.set_ylabel("Ranked candidate")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Predicted pIC$_{50}$")
    save_figure(fig, output, cfg)


def plot_admet_summary(top_hits_path: Path, output: Path, cfg: dict[str, Any]) -> None:
    data = pd.read_csv(top_hits_path)
    candidate_cols = [
        c
        for c in ["BBB_Martins", "HIA_Hou", "Bioavailability_Ma", "Ames", "hERG", "DILI", "bbb_score"]
        if c in data.columns
    ]
    if data.empty or not candidate_cols:
        return
    values = data[candidate_cols].apply(pd.to_numeric, errors="coerce")
    summary = values.median().dropna().sort_values()
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(cfg["figure_width_in"], max(2.5, 0.35 * len(summary))))
    ax.barh(summary.index, summary.values, color=NATURE_PALETTE[2], edgecolor="black", linewidth=0.4)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Median predicted endpoint value")
    ax.set_xlim(min(0, np.nanmin(summary.values) - 0.05), max(1, np.nanmax(summary.values) + 0.05))
    _despine(ax)
    save_figure(fig, output, cfg)



def plot_pareto_diversity(top_hits_path: Path, output: Path, cfg: dict[str, Any]) -> None:
    data = pd.read_csv(top_hits_path)
    required = {"activity_score", "admet_score", "final_score"}
    if data.empty or not required.issubset(data.columns):
        return
    x = pd.to_numeric(data["activity_score"], errors="coerce")
    y = pd.to_numeric(data["admet_score"], errors="coerce").fillna(0.5)
    final = pd.to_numeric(data["final_score"], errors="coerce")
    sizes = 18 + 55 * pd.to_numeric(data.get("qed", 0.5), errors="coerce").fillna(0.5).clip(0, 1)
    fig, ax = plt.subplots(figsize=(cfg["figure_width_in"], 3.2))
    scatter = ax.scatter(x, y, c=final, s=sizes, cmap="viridis", alpha=0.82, edgecolors="black", linewidths=0.35)
    if "pareto_rank" in data:
        front = pd.to_numeric(data["pareto_rank"], errors="coerce").fillna(99).eq(0)
        ax.scatter(x[front], y[front], facecolors="none", edgecolors=NATURE_PALETTE[1], s=sizes[front] + 30, linewidths=1.0, label="Pareto front")
        ax.legend(frameon=False, loc="lower left")
    ax.set_xlabel("Target activity score")
    ax.set_ylabel("ADMET desirability score")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Final score")
    _despine(ax)
    save_figure(fig, output, cfg)

def draw_top_structures(top_hits_path: Path, output: Path, top_n: int = 20) -> None:
    data = pd.read_csv(top_hits_path).head(top_n)
    if data.empty:
        return
    mols = [Chem.MolFromSmiles(s) for s in data["standardized_smiles"].astype(str)]
    legends = [
        f"{row.get('source_id', f'Hit {idx + 1}')}\nScore={row.get('final_score', float('nan')):.3f}"
        for idx, (_, row) in enumerate(data.iterrows())
    ]
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(450, 360),
        legends=legends,
        useSVG=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(600, 600))


def generate_all_figures(cfg: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    plot_cfg = cfg["plots"]
    configure_publication_style(plot_cfg)
    training_files = sorted(paths["processed"].glob("training_*.csv.gz"))
    outputs: list[str] = []
    plot_target_data_volume(training_files, paths["figures"] / "Fig01_target_data_volume", plot_cfg)
    plot_pic50_distributions(training_files, paths["figures"] / "Fig02_pic50_distributions", plot_cfg)
    plot_chemical_space(training_files, paths["figures"] / "Fig03_chemical_space", plot_cfg)
    plot_model_performance(paths["models"], paths["figures"] / "Fig04_model_performance", plot_cfg)
    plot_residuals(paths["models"], paths["figures"] / "Fig05_model_residuals", plot_cfg)
    plot_ensemble_weights(paths["models"], paths["figures"] / "Fig06_ensemble_weights", plot_cfg)
    for manifest_path in sorted(paths["screening"].glob("*/screening_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run = manifest_path.parent.name
        funnel = Path(manifest["funnel"])
        top_hits = Path(manifest["top_hits"])
        plot_screening_funnel(funnel, paths["figures"] / f"Fig07_screening_funnel_{run}", plot_cfg)
        if top_hits.exists() and top_hits.stat().st_size > 0:
            plot_top_hit_heatmap(top_hits, paths["figures"] / f"Fig08_target_heatmap_{run}", plot_cfg)
            plot_admet_summary(top_hits, paths["figures"] / f"Fig09_admet_summary_{run}", plot_cfg)
            plot_pareto_diversity(top_hits, paths["figures"] / f"Fig10_pareto_diversity_{run}", plot_cfg)
            draw_top_structures(top_hits, paths["figures"] / f"Fig11_top_structures_{run}.png")
    outputs.extend(str(p) for p in paths["figures"].glob("Fig*.*"))
    LOGGER.info("Generated %d figure files", len(outputs))
    return sorted(outputs)
