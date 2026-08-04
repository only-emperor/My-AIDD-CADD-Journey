# Validation report

Validation date: 2026-08-03

## Automated checks

- Python bytecode compilation: passed for `src` and `tests`.
- Unit and smoke tests: 12 passed.
- CLI loading/help: passed after editable installation.
- Local end-to-end smoke workflow: passed for a synthetic target dataset, including GNN training, checkpoint reuse, chunked screening, ranking, and figure generation.

## Covered behavior

- IC50 unit conversion and pIC50 calculation.
- Molecular standardization and descriptor generation.
- PAINS/Brenk/NIH structural-alert handling.
- Bemis–Murcko scaffold isolation.
- Molecular graph batching and edge-aware GNN forward pass.
- Multi-seed training smoke test.
- DrugBank XML stream parsing.
- Candidate input parsing, including whitespace-separated and compressed SMILES files.
- Pareto ranking and chemical-diversity clustering.
- ADMET/BBB rule evaluation and transparent missing-endpoint behavior.

## Environment limitation

The execution sandbox used to assemble this release blocked outbound DNS/network calls. Therefore the ChEMBL, PubChem, and ZINC clients were implemented and reviewed against their official API documentation, but live external API calls could not be completed in this sandbox. Run a small collection command in the target deployment environment before launching a large study.

## Scientific validation still required

Software tests do not validate biological claims. A publication-quality study still requires target-ID review, assay-level curation, temporal/external validation, repeated scaffold splits or bootstrapping, comparator models, applicability-domain analysis, prospective experimental assays, and expert review of structural alerts and ADMET predictions.
