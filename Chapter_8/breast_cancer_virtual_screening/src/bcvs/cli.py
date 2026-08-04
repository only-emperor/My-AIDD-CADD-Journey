from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from rich import print
from rich.console import Console
from rich.table import Table

from bcvs.config import ensure_project_layout, load_config
from bcvs.pipeline import collect_chembl, prepare_training_data, run_all, train_models
from bcvs.plots import generate_all_figures
from bcvs.screen import screen_library
from bcvs.sources.drugbank import import_drugbank_xml
from bcvs.sources.pubchem import PubChemClient
from bcvs.sources.zinc import ZINCClient
from bcvs.utils import configure_logging, write_dataframe

app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")
console = Console()
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _load(config: Path, verbose: bool = False):
    cfg = load_config(config)
    paths = ensure_project_layout(cfg)
    configure_logging(paths["logs"] / "bcvs.log", verbose=verbose)
    return cfg, paths


@app.command("init")
def init_project(
    destination: Annotated[Path, typer.Argument(help="Directory for a new project")],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    config_out = destination / "bcvs.yaml"
    if config_out.exists():
        raise typer.BadParameter(f"Already exists: {config_out}")
    shutil.copy2(DEFAULT_CONFIG, config_out)
    print(f"Created [bold]{config_out}[/bold]. Edit project.root and target settings before running.")


@app.command("collect-chembl")
def collect_chembl_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option("--force")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    cfg, _ = _load(config, verbose)
    outputs = collect_chembl(cfg, force=force)
    print_json(outputs)


@app.command("prepare")
def prepare_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option("--force")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    cfg, _ = _load(config, verbose)
    print_json(prepare_training_data(cfg, force=force))


@app.command("train")
def train_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option("--force")] = False,
    no_baseline: Annotated[bool, typer.Option("--no-baseline")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    cfg, _ = _load(config, verbose)
    print_json(train_models(cfg, force=force, include_baseline=not no_baseline))


@app.command("screen")
def screen_command(
    candidates: Annotated[Path, typer.Argument(help="CSV/TSV/SMI/SDF candidate library")],
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option("--force")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    cfg, paths = _load(config, verbose)
    print_json(screen_library(candidates, cfg, paths, force=force))


@app.command("plot")
def plot_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    cfg, paths = _load(config, verbose)
    files = generate_all_figures(cfg, paths)
    print_json({"figures": files})


@app.command("pubchem")
def pubchem_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    names_file: Annotated[Optional[Path], typer.Option("--names-file")] = None,
    cids_file: Annotated[Optional[Path], typer.Option("--cids-file")] = None,
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    cfg, _ = _load(config, verbose)
    client = PubChemClient(cfg["pubchem"])
    if names_file:
        names = [x.strip() for x in names_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        result = client.fetch_names(names)
    elif cids_file:
        cids = [int(x.strip().replace("CID", "")) for x in cids_file.read_text().splitlines() if x.strip()]
        result = client.fetch_cids(cids)
    else:
        raise typer.BadParameter("Provide --names-file or --cids-file")
    path = write_dataframe(result, output)
    print(f"Saved {len(result):,} PubChem records to [bold]{path}[/bold]")


@app.command("zinc")
def zinc_command(
    ids_file: Annotated[Path, typer.Option("--ids-file")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    configure_logging(verbose=verbose)
    ids = [x.strip() for x in ids_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    result = ZINCClient().fetch_ids(ids)
    path = write_dataframe(result, output)
    print(f"Saved {len(result):,} ZINC records to [bold]{path}[/bold]")


@app.command("drugbank")
def drugbank_command(
    xml: Annotated[Path, typer.Option("--xml", help="Licensed DrugBank XML export")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    configure_logging(verbose=verbose)
    path = import_drugbank_xml(xml, output)
    print(f"Saved licensed DrugBank import to [bold]{path}[/bold]")


@app.command("run-all")
def run_all_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
    candidates: Annotated[Optional[Path], typer.Option("--candidates")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    cfg, _ = _load(config, verbose)
    print_json(run_all(cfg, candidates, force=force))


@app.command("status")
def status_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG,
) -> None:
    import sqlite3

    cfg = load_config(config)
    paths = ensure_project_layout(cfg)
    db = paths["state"] / "pipeline.sqlite"
    if not db.exists():
        print("No pipeline state database yet.")
        raise typer.Exit()
    with sqlite3.connect(db) as con:
        rows = con.execute(
            "SELECT stage,status,datetime(started_at,'unixepoch'),datetime(finished_at,'unixepoch'),error FROM stages ORDER BY started_at"
        ).fetchall()
    table = Table("Stage", "Status", "Started (UTC)", "Finished (UTC)", "Error")
    for row in rows:
        table.add_row(*(str(x or "") for x in row))
    console.print(table)


def print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    app()
