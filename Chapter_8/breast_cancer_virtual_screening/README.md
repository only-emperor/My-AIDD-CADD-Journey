# B-CaVS: Auditable Breast-Cancer Virtual Screening

A restartable Python pipeline for target-specific breast-cancer small-molecule discovery. It collects ChEMBL IC50 records, converts them to pIC50, standardizes molecules, trains scaffold-split graph neural network ensembles, screens PubChem/ZINC/DrugBank or local libraries, applies physicochemical/PAINS/ADMET/BBB triage, preserves every filtering decision, and exports publication-grade figures.

> Research software only. Predictions are hypotheses for prioritization, not evidence of efficacy, safety, BBB penetration, or clinical suitability. Experimental validation and domain-expert review are required.

Terminology used here: the request’s “ADMEL” is interpreted as **ADMET**, and “PINAS” is interpreted as **PAINS** structural-alert filtering.

## 1. What “breast-cancer SMILES” means in this project

A disease does not have a SMILES string. The pipeline interprets the request as:

1. define breast-cancer-relevant molecular targets (default: ESR1, ERBB2, PIK3CA, AKT1, CDK4, CDK6, PARP1);
2. retrieve target-linked ChEMBL IC50 measurements and molecular structures;
3. train one target-specific pIC50 model per target;
4. predict and rank external candidate molecules across those targets.

Target mappings are resolved dynamically, written to `data/interim/targets_resolved.csv`, and must be reviewed before a publication. For frozen/reproducible studies, copy the reviewed `target_chembl_id` values into the YAML configuration.

## 2. Core workflow

```text
ChEMBL target search
        │
        ▼
IC50 activities ──► exact/standard record checks ──► pIC50 conversion
        │
        ▼
RDKit standardization ──► replicate aggregation ──► scaffold split
        │
        ▼
Pure-PyTorch molecular GIN models (multiple seeds)
        │
        ├── validation-RMSE ensemble weights
        ├── Morgan/ExtraTrees baseline
        └── test-set metrics and residuals
        │
        ▼
PubChem / ZINC / licensed DrugBank / local SMI-SDF-CSV candidates
        │
        ▼
Streaming standardization and cross-chunk deduplication
        │
        ▼
Physchem ─► PAINS/Brenk/NIH ─► GNN activity + uncertainty
        │
        ▼
Applicability domain ─► ADMET-AI ─► BBB prediction/annotation
        │
        ▼
Composite ranking + complete audit table + top hits + figures
```

## 3. Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate bcvs
pip install -e .
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[admet,parquet]"
```

`admet-ai` is optional at installation time but required for learned ADMET/BBB endpoints. Without it, the pipeline still runs and records missing ADMET endpoints; a transparent medicinal-chemistry BBB heuristic is reported as a fallback, never disguised as an experimental test.

## 4. Configure the study

Copy and edit the default configuration:

```bash
bcvs init my_project
# edit my_project/bcvs.yaml
```

Important fields:

- `project.root`: all raw data, models, logs, state, screening parts, and figures;
- `targets`: symbols, target weights, and optional pinned ChEMBL IDs;
- `chembl.min_target_samples`: minimum aggregated molecules for model training;
- `training.seeds`: ensemble members;
- `screening.activity_threshold_pic50`: activity triage threshold;
- `bbb.hard_filter`: set `true` only when BBB permeability is a project requirement, such as a brain-metastasis/CNS program;
- `admet.rules`: endpoint-specific direction, threshold, and whether a failure is hard or advisory;
- `structural_alerts`: PAINS/Brenk/NIH behavior.

The default config is deliberately conservative but is not a universal medicinal-chemistry truth. Thresholds must be justified for the specific target, assay system, route of administration, tissue, and intended product profile.

## 5. Run the pipeline

### Stepwise execution

```bash
bcvs collect-chembl -c my_project/bcvs.yaml
bcvs prepare        -c my_project/bcvs.yaml
bcvs train          -c my_project/bcvs.yaml
bcvs screen candidates.smi -c my_project/bcvs.yaml
bcvs plot           -c my_project/bcvs.yaml
bcvs status         -c my_project/bcvs.yaml
```

### One command

```bash
bcvs run-all -c my_project/bcvs.yaml --candidates candidates.smi
```

Re-running the same command reuses completed stages and screening chunks. Use `--force` to intentionally recompute a stage.

### Quick smoke input

```bash
bcvs screen scripts/demo_candidates.smi -c my_project/bcvs.yaml
```

## 6. Data-source adapters

### ChEMBL

The pipeline uses the official REST API, resolves targets, downloads IC50 activity pages with retries, and preserves raw JSON pages. pIC50 is calculated as:

```text
pIC50 = −log10(IC50 in mol/L)
```

Examples: `10 nM → 8.0`; `1 µM → 6.0`.

Rows are excluded from regression by configurable rules such as missing structures, nonpositive values, unsupported units, censored relations, nonstandard records, potential duplicates, data-validity flags, and disallowed assay types. The exclusion table is retained.

### PubChem

Small identifier/name queries use PUG REST with throttling below PubChem’s stated five-request-per-second ceiling:

```bash
bcvs pubchem --names-file scripts/pubchem_names.txt \
  --output runs/pubchem_known_drugs.csv -c my_project/bcvs.yaml

bcvs pubchem --cids-file my_cids.txt \
  --output runs/pubchem_candidates.csv -c my_project/bcvs.yaml
```

For large PubChem screens, download an official bulk SDF/SMI/CSV shard and pass it directly to `bcvs screen`. PUG REST should not be abused as a whole-database crawler.

### ZINC

Exact-ID lookup is provided for small lists:

```bash
bcvs zinc --ids-file zinc_ids.txt --output zinc_lookup.csv
```

For millions of ZINC compounds, use an official downloaded shard and stream it through `bcvs screen`. Website layouts and anti-bot pages are not a stable bulk-data API.

### DrugBank

DrugBank content requires a license. This repository does **not** scrape or bypass access controls. It parses an XML export that the user is authorized to use:

```bash
bcvs drugbank --xml full_database.xml --output drugbank_authorized.csv.gz
bcvs screen drugbank_authorized.csv.gz -c my_project/bcvs.yaml
```

The import retains DrugBank ID, name, groups, SMILES, InChIKey, and listed target names when available.

## 7. Model design

### Dataset curation

- exact IC50 relations by default;
- molar-unit conversion to pIC50;
- RDKit cleanup, largest-fragment selection, optional uncharging and tautomer canonicalization;
- standardized InChIKey/SMILES deduplication;
- replicate median (or mean) with replicate count, range, and standard deviation;
- highly discordant replicate groups excluded by configurable SD threshold.

### Validation

Random molecular splits often inflate performance through scaffold leakage. This project uses Bemis–Murcko scaffold splitting by default and stores row-level split assignments. Each target model reports RMSE, MAE, and R² for training, validation, and held-out test data.

### GNN

The model is a graph isomorphism network implemented directly in PyTorch:

- atoms are graph nodes with categorical embeddings;
- covalent bonds define bidirectional message-passing edges;
- edge-aware GINE-style residual layers use atom messages plus bond type, conjugation, ring, and stereochemical features;
- global mean pooling produces a molecule embedding;
- a regression head predicts pIC50;
- Smooth L1 loss, AdamW, learning-rate reduction, gradient clipping, and early stopping are used.

No PyTorch Geometric installation is required.

### Ensemble weights

Each random-seed model receives a softmax weight derived from validation RMSE. The manifest stores every checkpoint, validation metric, best epoch, and final weight. Candidate uncertainty is the weighted standard deviation across ensemble members.

### Baseline

A Morgan-fingerprint ExtraTrees model is trained as a non-neural baseline. A credible paper should compare the GNN against this baseline and report confidence intervals from repeated splits or bootstrapping rather than relying on one favorable split.

## 8. Screening logic and audit trail

Each output part retains original columns plus:

- raw and standardized SMILES, InChIKey, parse/standardization status;
- duplicate flag;
- MW, logP, TPSA, HBD, HBA, rotatable bonds, formal charge, QED, synthetic-accessibility score, Lipinski violations;
- PAINS, Brenk, and NIH alert counts and descriptions;
- target-specific predicted pIC50 and ensemble uncertainty;
- target-specific activity score;
- target-specific maximum training-set Tanimoto similarity;
- ADMET-AI endpoint columns when available;
- ADMET hard-pass, advisory score, and reasons;
- learned or heuristic BBB score, endpoint name, and pass flag;
- applicability-domain pass, novelty score, final score, final-pass flag;
- final rank, ECFP4/Butina diversity cluster, cluster size, Pareto rank, and representative flag for final hits;
- semicolon-delimited rejection reason.

A molecule is not silently deleted. Rejected rows remain in compressed chunk files under:

```text
<project.root>/screening/<library_name>/parts/
```

The final directory contains:

- `screening_manifest.json`: input checksum, part paths, model targets, ADMET status, rejection counts;
- `screening_funnel_by_chunk.csv`: stage counts for every chunk;
- `top_hits.csv`: deduplicated, Pareto-annotated, ECFP4/Butina-diversified top-ranked molecules;
- `dedupe.sqlite`: persistent cross-chunk InChIKey registry.

## 9. Checkpoint/restart behavior

`state/pipeline.sqlite` uses WAL mode and records stage/chunk states:

- running;
- done;
- failed;
- input hash;
- output path;
- metadata/error.

Completed chunks are skipped on rerun. Screening deduplication is persisted separately. Raw ChEMBL JSON pages are also cached, which supports provenance and troubleshooting.

## 10. Publication-grade figures

`bcvs plot` writes PNG (600 dpi), editable SVG, and vector PDF. Text is English and configured for Times New Roman with Times/DejaVu Serif fallback. Figures include:

1. target data volume;
2. pIC50 distributions;
3. Morgan-fingerprint chemical-space PCA;
4. observed-versus-predicted held-out performance;
5. residual diagnostics;
6. ensemble weights;
7. screening funnel;
8. target prediction heatmap for top hits;
9. ADMET/BBB endpoint summary;
10. activity–ADMET Pareto/diversity view;
11. top-hit molecular structure grid.

“Nature-level” cannot be guaranteed by a plotting style alone. Publication quality also requires correct statistics, clear legends, biological rationale, assay-context discussion, uncertainty, external validation, and journal-specific final sizing. The generated figures are an editable starting point with consistent typography, line weights, and vector outputs.

## 11. Scaling to PubChem/ZINC size

For very large libraries:

1. use official bulk files rather than per-molecule web requests;
2. screen one shard at a time or use a shared project directory with unique filenames;
3. reduce `screening.chunk_size` if RAM is limited;
4. run cheap standardization/physchem/alerts before GNN and ADMET, as implemented;
5. reserve ADMET-AI for the post-GNN subset when throughput is the bottleneck;
6. preserve each shard’s manifest and combine only the per-shard top hits;
7. use a GPU for GNN/ADMET inference, but test CPU/GPU numerical consistency.

The code deliberately does not concatenate every million-row part into one in-memory DataFrame. Full audits remain in chunk files; only passing top candidates are merged for ranking.

## 12. Scientific cautions

- ChEMBL IC50 values mix laboratories, constructs, assay formats, readouts, and experimental conditions. Target-specific curation is still required.
- pIC50 conversion makes units comparable; it does not remove inter-assay bias.
- PAINS alerts are triage flags, not proof that a molecule is invalid. Inspect mechanisms and assay context.
- BBB permeability is context-dependent. Peripheral breast-cancer drugs do not universally need BBB penetration; brain-metastasis programs may.
- ADMET-AI outputs are model predictions and may be outside their applicability domains.
- Maximum Tanimoto similarity is a simple applicability indicator, not a formal uncertainty guarantee.
- A high composite score must never be interpreted as clinical safety or efficacy.
- External prospective validation is essential for SCI-level claims.

## 13. Reproducibility checklist for a paper

- pin ChEMBL release/date and reviewed target IDs;
- archive configuration and environment lock file;
- report every exclusion rule and attrition count;
- report scaffold split and random seeds;
- compare against fingerprint baselines;
- add confidence intervals/bootstraps and repeated splits;
- perform Y-randomization or permutation controls;
- assess applicability domain and calibration;
- validate top hits with orthogonal assays;
- disclose database licenses and access dates;
- provide raw/processed data subject to redistribution terms;
- include model cards and failure analysis.

## 14. Official resources used by the implementation

- ChEMBL REST web services: `https://www.ebi.ac.uk/chembl/api/data/docs`
- PubChem PUG REST: `https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest`
- ZINC/Docking.org documentation: `https://wiki.docking.org/index.php/Category:ZINC`
- DrugBank data packages/licensing: `https://go.drugbank.com/data_packages`
- RDKit FilterCatalog: `https://www.rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html`
- ADMET-AI: `https://github.com/swansonk14/admet_ai`

## 15. Tests

```bash
pytest
python -m compileall -q src
```

The tests cover pIC50 conversion, standardization, scaffold isolation, graph batching/model forward pass, structural filters, DrugBank XML parsing, and a small GNN training smoke test.
