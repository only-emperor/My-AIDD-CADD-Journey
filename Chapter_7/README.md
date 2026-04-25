# Sweetness Prediction from SMILES: Bi-LSTM vs Transformer 🧪💊

## 📖 Overview
This repository provides a complete, reproducible deep learning pipeline for predicting molecular sweetness (`logSw`) directly from SMILES strings. We implement and compare two state-of-the-art sequence models—**Bi-LSTM** and **Transformer**—to learn chemical structural patterns and perform regression tasks. The project includes full data preprocessing, custom SMILES tokenization, model training/evaluation, and rich visualization for performance analysis.

## ✨ Key Features

### 1. Custom SMILES Tokenization
- Regex-based tokenizer tailored for molecular SMILES grammar (supports atoms, bonds, cycles, and special symbols)
- Auto vocabulary construction with special tokens (`<pad>`, `<unk>`, `<sos>`, `<eos>`)
- Vocabulary persistence (save/load via JSON)
- Fixed-length sequence padding and encoding/decoding

### 2. Standard Dataset Pipeline
- Automated CSV data cleaning (missing value removal, column normalization)
- 80%/10%/10% train/validation/test random split
- Encapsulated PyTorch `Dataset` and `DataLoader` for batch processing

### 3. Dual Model Architectures

| Model       | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| Bi-LSTM     | Bidirectional LSTM with embedding layer, dropout, and MLP regression head   |
| Transformer | Transformer encoder with sinusoidal positional encoding, multi-head self-attention, and masked mean pooling |

### 4. Training & Evaluation
- Unified training loop for both models
- MSE loss + Adam optimizer + gradient clipping
- Regression metrics: MSE and R² score
- One-click model weight saving

### 5. Rich Visualization
- Dataset dashboard (sequence length, token frequency, label distribution)
- Learning curves (train/val loss)
- True vs. predicted scatter plots
- Head-to-head model comparison (val loss & R²)

## 🛠️ Requirements
Install dependencies via pip:
```bash
pip install numpy pandas torch matplotlib seaborn scikit-learn
