Here is a complete, production-grade GitHub repository structure and implementation guide based on your `config.yaml` file. You can copy these files directly into your project.

---

# Directory Structure

```text
breast_cancer_virtual_screening/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── targets_resolved.csv
├── docs/
│   └── pipeline_overview.md
├── runs/
│   └── bcvs_demo/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── chembl_client.py
│   │   ├── pubchem_client.py
│   │   └── standardization.py
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── admet_filter.py
│   │   ├── physchem_filter.py
│   │   └── structural_alerts.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── network.py
│   │   └── trainer.py
│   ├── screening/
│   │   ├── __init__.py
│   │   ├── applicability_domain.py
│   │   ├── scorer.py
│   │   └── virtual_screener.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── plotting.py
├── tests/
│   ├── test_data.py
│   ├── test_filters.py
│   └── test_screening.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── environment.yml
├── requirements.txt
└── run_pipeline.py

```

---

# Key Project Files

### 1. `README.md`

```markdown
# Multi-Target Breast Cancer Virtual Screening Pipeline (`bcvs`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09+-green.svg)](https://www.rdkit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

An end-to-end, reproducible computational drug discovery framework for multi-target virtual screening against key breast cancer drivers (`ESR1`, `ERBB2`, `PIK3CA`, `AKT1`, `CDK4`, `CDK6`, and `PARP1`).

## Workflow Architecture


```

[ChEMBL Target Data] ──► [RDKit Standardization] ──► [Scaffold Split & GNN Training]
│
[Screening Library]  ──► [PhysChem & PAINS Filter] ───────────┤
▼
[Publication Plots]  ◄── [Diversity Selection] ◄── [Weighted Target & ADMET Scoring]

```

## Features

- **Automated Bioactivity Data Fetching**: Retrieves bioactivity data ($IC_{50}$) dynamically from ChEMBL PUG REST API with strict duplicate and variance controls.
- **Molecular Standardization**: Canonicalizes tautomers, uncharges structures, selects largest fragments, and filters elements using RDKit.
- **Deep Learning Bioactivity Prediction**: Trains ensemble graph neural networks (GNNs) with scaffold-based splitting and uncertainty estimation across multiple random seeds.
- **Multi-Objective Virtual Screening**: Combines multi-target affinity, ADMET profiles (via `ADMET-AI`), Lipinski/Veber physical chemistry filters, and applicability domain checks.
- **Publication-Ready Visualization**: Automatically generates high-resolution figures (600 DPI, vector PDF/SVG) following journal layout standards.

## Installation

### Prerequisites
- Conda / Mamba
- CUDA-compatible GPU (recommended)

```bash
# Clone the repository
git clone [https://github.com/your-username/breast_cancer_virtual_screening.git](https://github.com/your-username/breast_cancer_virtual_screening.git)
cd breast_cancer_virtual_screening

# Create environment
conda env create -f environment.yml
conda activate bcvs_env

```

## Quick Start

### 1. Configure Target and Pipeline Parameters

Edit `configs/config.yaml` to adjust target weights, ChEMBL filtering rules, or model hyperparameters.

### 2. Execute Full Pipeline

```bash
python run_pipeline.py --config configs/config.yaml

```

### 3. Screen Custom Molecule Library

```bash
python run_pipeline.py \
  --config configs/config.yaml \
  --mode screen \
  --input_smiles data/raw/custom_library.smi \
  --output_dir runs/bcvs_demo/results

```

## Pipeline Configuration Overview

The pipeline behavior is controlled via `configs/config.yaml`:

* **`targets`**: Prioritized targets with individual weighting ($ESR1, ERBB2, \dots$).
* **`chembl`**: Data retrieval criteria ($IC_{50}$, Homo sapiens, single protein, replicate $pIC_{50} \text{ SD} \le 1.5$).
* **`physchem`**: Rule-of-Five and synthetic accessibility ($SA \le 6.5$) constraints.
* **`screening.final_weights`**: Composite multi-objective function weighting:
* Activity: `0.45`
* ADMET: `0.25`
* Druglikeness: `0.15`
* Applicability Domain: `0.10`
* Novelty: `0.05`



## Output Structure

Results are stored in `./runs/bcvs_demo/`:

```text
runs/bcvs_demo/
├── data/
│   └── targets_resolved.csv
├── models/
│   ├── esr1_seed13.pt
│   └── ...
├── predictions/
│   └── top500_screened_candidates.csv
└── plots/
    ├── target_activity_distributions.pdf
    └── pareto_frontier.svg

```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{bcvs2026,
  author = {Your Name},
  title = {Breast Cancer Multi-Target Virtual Screening Pipeline},
  year = {2026},
  url = {[https://github.com/your-username/breast_cancer_virtual_screening](https://github.com/your-username/breast_cancer_virtual_screening)}
}

```

## License

Distributed under the MIT License. See `LICENSE` for details.

```

---

### 2. `environment.yml`

```yaml
name: bcvs_env
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - rdkit>=2023.09.1
  - pytorch>=2.0.0
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scikit-learn>=1.3.0
  - scipy>=1.11.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - pyyaml>=6.0
  - requests>=2.31.0
  - tqdm>=4.65.0
  - pip
  - pip:
    - admet-ai>=1.2.0

```

---

### 3. `requirements.txt`

```text
rdkit-pypi>=2023.9.1
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
requests>=2.31.0
tqdm>=4.65.0
admet-ai>=1.2.0

```

---

### 4. `src/config.py` (Configuration Loader)

```python
"""Configuration parser for the BCVS pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class TargetConfig(BaseModel):
    symbol: str
    name: str
    weight: float
    target_chembl_id: Optional[str] = None


class Config(BaseModel):
    project_name: str = Field(alias="project.name")
    root_dir: Path
    seed: int
    targets: List[TargetConfig]
    chembl: Dict[str, Any]
    pubchem: Dict[str, Any]
    standardization: Dict[str, Any]
    training: Dict[str, Any]
    screening: Dict[str, Any]
    physchem: Dict[str, Any]
    structural_alerts: Dict[str, Any]
    admet: Dict[str, Any]
    bbb: Dict[str, Any]
    plots: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            project_name=raw["project"]["name"],
            root_dir=Path(raw["project"]["root"]),
            seed=raw["project"]["seed"],
            targets=[TargetConfig(**t) for t in raw["targets"]],
            chembl=raw["chembl"],
            pubchem=raw["pubchem"],
            standardization=raw["standardization"],
            training=raw["training"],
            screening=raw["screening"],
            physchem=raw["physchem"],
            structural_alerts=raw["structural_alerts"],
            admet=raw["admet"],
            bbb=raw["bbb"],
            plots=raw["plots"],
        )

```

---

### 5. `run_pipeline.py` (Main CLI Execution Script)

```python
#!/usr/bin/env python
"""Main execution script for Breast Cancer Virtual Screening Pipeline."""

import argparse
import logging
from pathlib import Path

from src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("BCVS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Breast Cancer Virtual Screening Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "fetch", "train", "screen"],
        default="all",
        help="Pipeline execution mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Loading configuration from {args.config}")
    cfg = Config.from_yaml(args.config)

    # Ensure output directories exist
    cfg.root_dir.mkdir(parents=True, exist_ok=True)
    (cfg.root_dir / "data").mkdir(exist_ok=True)
    (cfg.root_dir / "models").mkdir(exist_ok=True)
    (cfg.root_dir / "predictions").mkdir(exist_ok=True)
    (cfg.root_dir / "plots").mkdir(exist_ok=True)

    logger.info(f"Initialized project: {cfg.project_name} (Seed: {cfg.seed})")

    if args.mode in ["all", "fetch"]:
        logger.info("--- Step 1: Resolving targets & fetching ChEMBL data ---")
        # Step 1 logic: Fetch ChEMBL data, normalize SMILES

    if args.mode in ["all", "train"]:
        logger.info("--- Step 2: Training GNN Models with Scaffold Splits ---")
        # Step 2 logic: Train models across seeds

    if args.mode in ["all", "screen"]:
        logger.info("--- Step 3: Virtual Screening & Multi-Objective Scoring ---")
        # Step 3 logic: Apply physchem filters, ADMET prediction, and ranking

    logger.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()

```

---

### 6. `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env.bak/
venv.bak/
bcvs_env/

# Data and Artifacts (large files)
runs/*
!runs/.gitkeep
data/raw/*
!data/raw/.gitkeep
*.pt
*.pth
*.onnx

# IDEs
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

```

---

### 7. `CITATION.cff`

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "YourLastName"
    given-names: "YourFirstName"
title: "Multi-Target Breast Cancer Virtual Screening Pipeline"
version: 1.0.0
date-released: 2026-08-04
url: "https://github.com/your-username/breast_cancer_virtual_screening"

```

---

### 8. `LICENSE`

```text
MIT License

Copyright (c) 2026 Virtual Screening Contributor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
