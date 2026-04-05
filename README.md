# 🧬 My AIDD & CADD Journey 

Welcome to my personal repository documenting my exploration, projects, and research at the intersection of **Artificial Intelligence in Drug Discovery (AIDD)**, **Computer-Aided Drug Design (CADD)**, and **Medical Image Analysis**.

![Python](https://img.shields.io/badge/Language-Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C.svg) ![RDKit](https://img.shields.io/badge/Cheminformatics-RDKit-green.svg) ![Radiomics](https://img.shields.io/badge/Domain-Medical_Imaging-red.svg) ![Machine Learning](https://img.shields.io/badge/AI-Machine_Learning-orange.svg)

## 🎯 About This Repository

The goal of this repository is to build and share end-to-end pipelines that accelerate scientific discovery. Here, you will find code implementation ranging from **molecular property prediction (QSAR)** to **deep radiomics feature extraction** from medical imaging. 

I strongly believe in not just building "black-box" models, but focusing on **Explainable AI (XAI)** and robust feature engineering to derive true biological and chemical insights.

---

## 📂 Project Portfolio

### 1. 💊 Cheminformatics & QSAR (AIDD)
*Applying machine learning to chemical space and molecular representations.*
* **[SweetPred-ML: Molecular Sweetness Prediction](./Link-To-Your-Folder)**
  * **Description**: An end-to-end QSAR pipeline to predict molecular sweetness (`logSw`) directly from SMILES strings.
  * **Highlights**: 
    * 1D/2D Descriptor & Morgan Fingerprint extraction using `RDKit`.
    * Chemical space mapping via `t-SNE`.
    * High-performance modeling with `LightGBM`.
    * **Explainable AI**: Global and local interpretability using `SHAP` waterfall and summary plots.

### 2. 🧠 Medical Image Analysis & Deep Radiomics
*Quantifying tumor heterogeneity through hybrid feature extraction.*
* **[CT Radiomics & Fractal Analysis](./Link-To-Your-Folder)**
  * **Description**: A comprehensive toolkit for extracting advanced quantitative biomarkers from CT/MRI NIfTI images.
  * **Highlights**:
    * Handcrafted Radiomics: `pyradiomics`, Wavelet transforms, and Fast Fourier Transform (FFT) analysis.
    * Structural Complexity: 2D/3D **Fractal Dimension (Box-counting)** calculations.
    * **Deep Radiomics**: Leveraging pre-trained deep neural networks (**ResNet-50** & **2D MaxViT**) to extract high-dimensional spatial and attention-based embeddings.

*(More projects will be added as my journey continues...)*

---

## 🛠️ Tech Stack & Tools

| Category | Libraries & Frameworks |
| :--- | :--- |
| **Cheminformatics** | `RDKit`, `ChemPy` |
| **Medical Imaging** | `SimpleITK`, `nibabel`, `pyradiomics` |
| **Deep Learning** | `PyTorch`, `torchvision`, `timm` (Vision Transformers) |
| **Machine Learning** | `scikit-learn`, `LightGBM`, `XGBoost` |
| **Data & XAI** | `pandas`, `numpy`, `SHAP`, `scipy` |
| **Visualization** | `matplotlib`, `seaborn` |

---

## 🚀 How to Navigate

Feel free to explore the subdirectories. Each project folder contains its own detailed `README.md`, Jupyter Notebooks (`.ipynb`), or Python scripts (`.py`), along with instructions on how to set up the environment and run the code.

```bash
# Clone the repository
git clone https://github.com/only-emperor/My-AIDD-CADD-Journey.git
cd My-AIDD-CADD-Journey
