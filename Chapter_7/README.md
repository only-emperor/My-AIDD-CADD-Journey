Sweetness Prediction from SMILES: Bi-LSTM vs Transformer 🧪💊








📖 Overview
This repository provides a complete, reproducible deep learning pipeline for predicting molecular sweetness (logSw) directly from SMILES strings. We implement and compare two state-of-the-art sequence models—Bi-LSTM and Transformer—to learn chemical structural patterns and perform regression tasks. The project includes full data preprocessing, custom SMILES tokenization, model training/evaluation, and rich visualization for performance analysis.
✨ Key Features
1. Custom SMILES Tokenization
Regex-based tokenizer tailored for molecular SMILES grammar (supports atoms, bonds, cycles, and special symbols)
Auto vocabulary construction with special tokens (<pad>, <unk>, <sos>, <eos>)
Vocabulary persistence (save/load via JSON)
Fixed-length sequence padding and encoding/decoding
2. Standard Dataset Pipeline
Automated CSV data cleaning (missing value removal, column normalization)
80%/10%/10% train/validation/test random split
Encapsulated PyTorch Dataset and DataLoader for batch processing
3. Dual Model Architectures
表格
Model	Description
Bi-LSTM	Bidirectional LSTM with embedding layer, dropout, and MLP regression head
Transformer	Transformer encoder with sinusoidal positional encoding, multi-head self-attention, and masked mean pooling
4. Training & Evaluation
Unified training loop for both models
MSE loss + Adam optimizer + gradient clipping
Regression metrics: MSE and 
R 
2
 
 score
One-click model weight saving
5. Rich Visualization
Dataset dashboard (sequence length, token frequency, label distribution)
Learning curves (train/val loss)
True vs. predicted scatter plots
Head-to-head model comparison (val loss & 
R 
2
 
)
🛠️ Requirements
Install dependencies via pip:
bash
运行
pip install numpy pandas torch matplotlib seaborn scikit-learn
Note: GPU is recommended for faster model training.
📁 Project Structure
plaintext
sweetness-prediction/
├── sweetness_prediction.py  # Main pipeline script
├── SweetpredDB.csv          # Your dataset (place here)
├── vocab.json               # Auto-generated SMILES vocabulary
├── bilstm.pt                # Trained Bi-LSTM weights
├── transformer.pt           # Trained Transformer weights
└── README.md                # This file
🚀 Quick Start
1. Prepare Data
Place your dataset (named SweetpredDB.csv) in the project root. The CSV must contain two columns:
Smiles: Molecular SMILES strings
logSw: Sweetness target value (logarithm scale)
2. Run the Pipeline
Execute the main script to start training and evaluation:
bash
运行
python sweetness_prediction.py
3. Check Outputs
Trained model weights: bilstm.pt, transformer.pt
Vocabulary file: vocab.json
Interactive plots (dataset stats, learning curves, model comparison)
📊 Expected Results
表格
Model	Best Val 
R 
2
 
Test MSE
Bi-LSTM	~0.75 (example)	~0.45 (example)
Transformer	~0.78 (example)	~0.40 (example)
Note: Actual results depend on your dataset size and quality.
📜 License
This project is licensed under the MIT License. Feel free to use it for research or commercial purposes.
🙏 Acknowledgments
Inspired by cheminformatics and molecular property prediction research
Built with PyTorch, pandas, and scikit-learn
