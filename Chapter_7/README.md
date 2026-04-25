SMILES Sweetness Prediction: Bi-LSTM vs. Transformer 🍭🧪🤖
PythonLicenseChemoinformaticsPyTorch

📖 Overview
This repository contains a deep learning pipeline for predicting the sweetness intensity of molecules (logSw) directly from their SMILES (Simplified Molecular Input Line Entry System) strings. The project compares two powerful architectures—Bidirectional LSTM (RNN) and Transformer (Attention)—to determine which is more effective at capturing the structural dependencies of sweet-tasting compounds.

✨ Key Features
Custom SMILES Tokenizer:

Uses a specialized regex-based tokenizer to correctly identify atoms (e.g., Br, Cl), rings, and bonds.
Includes special tokens (<sos>, <eos>, <pad>) for sequence processing.
Exploratory Data Analysis (EDA) Dashboard:

Visualizes sequence length distributions.
Analyzes top-10 most frequent chemical tokens.
Plots the statistical distribution of the logSw target variable.
Dual Model Architecture Comparison:

Bi-LSTM Regressor: A sequence-based model using bidirectional Long Short-Term Memory units to capture local structural context.
Transformer Regressor: A state-of-the-art model using Multi-Head Self-Attention and Positional Encoding to capture global molecular dependencies.
Advanced Evaluation:

Real-time training/validation loss tracking.
Performance metrics using 
R
2
R 
2
  Score and MSE.
Direct model "Showdown" visualizations comparing learning curves and prediction accuracy.
🏗️ Model Architectures
1. Bi-LSTM Regressor
Embedding Layer: Maps tokens to a 128-dimensional space.
LSTM: 2-layer bidirectional processing with Dropout.
Pooling: Mean pooling across the sequence dimension for a fixed-size molecular representation.
2. Transformer Regressor
Positional Encoding: Injects sequence order information into the embeddings.
Encoder: 3-layer Transformer Encoder with 4 attention heads.
Masked Mean Pooling: Aggregates features while ignoring padding tokens for cleaner signal propagation.
🛠️ Requirements
Ensure you have a GPU-enabled environment (like Google Colab) for faster training.

Bash

pip install pandas numpy torch matplotlib seaborn scikit-learn
🚀 Getting Started
Prepare Data: Ensure your dataset is in CSV format with columns Smiles and logSw.
Run Pipeline:
Python

# The script automatically handles:
# 1. Tokenization and Vocab building
# 2. Data splitting (Train/Val/Test)
# 3. Model training for both architectures
# 4. Result plotting and model saving (.pt files)
python sweetness_prediction.py
📊 Results & Visualization
The pipeline generates several plots:

Dataset Dashboard: Understanding the chemical composition of your data.
Learning Curves: Monitoring for overfitting or convergence issues.
Regression Plots: Predicted vs. Actual values to visualize 
R
2
R 
2
  performance.
Model Showdown: A direct overlay comparison of which model minimizes MSE more effectively.
🚧 Project Status / Future Updates
 Integration of Pre-trained ChemBERTa embeddings.
 Addition of Attention Map visualizations to see which atoms contribute most to "sweetness".
 Support for Data Augmentation (SMILES randomization).
