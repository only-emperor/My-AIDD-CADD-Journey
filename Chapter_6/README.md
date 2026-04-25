# Sweetness Prediction using Descriptors, LightGBM & SHAP 🍬📊

## 📖 Overview
This repository provides an end-to-end, interpretable machine learning pipeline for predicting molecular sweetness (`logSw`) from SMILES strings. Unlike deep sequence models, this project leverages **domain‑aware molecular descriptors** (calculated with RDKit) and a **LightGBM regressor**, followed by in‑depth **SHAP analysis** to explain individual predictions. The pipeline is designed to be run in a Colab environment but can be easily adapted to any local Python setup.

## ✨ Key Features
- **Robust feature extraction** – calculates a curated set of 2D molecular descriptors and 2048‑bit Morgan fingerprints per molecule.
- **Data‑driven preprocessing** – variance filtering, missing value imputation, and train/test splitting.
- **LightGBM training with early stopping** – prevents overfitting using multiple evaluation sets and callbacks.
- **Comprehensive evaluation** – RMSE, R², regression plot with perfect‑fit and regression lines.
- **SHAP model explainability** – global summary plots, waterfall plots for single molecules, and molecular structure visualisation.
- **Persistence** – model and preprocessing info saved for later inference (`.pkl`).

## 🛠️ Requirements
Install the main dependencies (if not already present):
```bash
pip install rdkit-pypi seaborn shap lightgbm xgboost lazypredict
# Note: xgboost and lazypredict are imported but not used in the core pipeline; you can drop them if desired.
