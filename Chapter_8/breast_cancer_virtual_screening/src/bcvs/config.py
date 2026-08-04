from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    cfg = deepcopy(cfg)
    root = Path(cfg["project"]["root"]).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    cfg["project"]["root"] = str(root)
    cfg["_config_path"] = str(path)
    return cfg


def ensure_project_layout(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["project"]["root"])
    paths = {
        "root": root,
        "raw": root / "data" / "raw",
        "interim": root / "data" / "interim",
        "processed": root / "data" / "processed",
        "candidates": root / "data" / "candidates",
        "models": root / "models",
        "screening": root / "screening",
        "figures": root / "figures",
        "logs": root / "logs",
        "cache": root / "cache",
        "state": root / "state",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
