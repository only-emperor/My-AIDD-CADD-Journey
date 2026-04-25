Sweetness Prediction from SMILES: Bi-LSTM vs Transformer 🧪💊
PythonDeep LearningCheminformaticsSMILES Regression
📖 OverviewThis repository provides a comprehensive pipeline for molecular sweetness prediction based on SMILES string sequence modeling. It integrates custom SMILES tokenization, dataset processing, Bi-LSTM and Transformer deep learning architectures, end-to-end training, quantitative evaluation and visual analysis. This work realizes molecular structure-driven logSw regression, comparing sequential feature extraction capabilities between recurrent neural networks and attention-based models.
✨ Key FeaturesCustom SMILES Tokenizer:
Regular expression-based molecular tokenization for atoms, bonds, brackets and special chemical symbols.Auto-build vocabulary, support <pad>, <unk>, <sos>, <eos> special tokens.Vocabulary serialization and reversible encode/decode function.
Standard Dataset Pipeline:
Automatic CSV data cleaning, missing value filtering and column normalization.80%/10%/10% train/val/test random split with fixed random seed.PyTorch Dataset & DataLoader encapsulation for batch training.
Bi-LSTM Regression Model:
Bidirectional LSTM structure for local sequence feature capture of molecules.Embedding layer, bidirectional recurrent module and dropout regularization.Global mean pooling + MLP regression head for continuous logSw prediction.
Transformer Regression Model 🆕:
Sinusoidal positional encoding to inject sequence position information.Multi-head self-attention encoder to capture long-range chemical dependencies.Padding mask mechanism and masked mean pooling for variable-length SMILES.
Visualization & Quantitative Evaluation:
Dataset distribution analysis: sequence length, token frequency and label distribution.Learning curve monitoring for training and validation loss.True-predicted scatter plot and R² metric calculation.Dual-model horizontal comparison of validation loss and R² performance.
🛠️ RequirementsInstall the necessary dependencies. (Note: GPU is recommended for faster Transformer and LSTM training).
bash
运行
pip install numpy pandas torch matplotlib seaborn scikit-learn
🚧 Project Status / Future Updates
 Complete SMILES tokenization and vocabulary construction
 Bi-LSTM and Transformer dual model regression framework
 Unified training loop, evaluation metrics and model saving
 Multi-dimensional data visualization and model comparison
 Add Transformer attention weight visualization
 Introduce molecular fingerprint hybrid feature fusion
 Support cross-validation and hyperparameter search
 Add single SMILES online inference function
 Optimize model structure and regularization strategy
