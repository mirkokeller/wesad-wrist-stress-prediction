# 🧠 Wrist-only WESAD Stress Prediction

A machine learning pipeline designed to predict human stress levels using **wrist-only biosignals** from the publicly available **WESAD (Wrist and Chest Affect Detection)** dataset.

This repository utilizes non-invasive signals collected from the wrist, omitting chest-worn sensors.

---

## 🌟 Key Features

*   **Wrist-only Biosignals**: Predicts emotional state (stress vs. baseline vs. amusement) using:
    *   **BVP** (Blood Volume Pulse - Photoplethysmography)
    *   **EDA** (Electrodermal Activity - Skin Conductance)
    *   **TEMP** (Skin Temperature)
    *   **ACC** (3-axis Accelerometer)
*   **Leave-One-Subject-Out (LOSO) Cross-Validation**: Implements subject-independent cross-validation to prevent data leakage.
*   **Biosignal Feature Extraction**: Processes raw signals using rolling windows (e.g., 60 seconds with a 1-second step) to extract statistical features, Heart Rate Variability (HRV) from BVP, and Skin Conductance Responses (SCR) from EDA.
*   **Multi-Model Benchmarking**: Compares performances across various classifiers including Random Forests, Support Vector Machines (SVM), and XGBoost.

---

## 📂 Repository Structure

```
├── data/                     # Data directories (Git Ignored)
│   ├── 01_raw/               # Raw WESAD dataset files
│   ├── 02_intermediate/      # Preprocessed signal windows
│   └── 03_processed/         # Final feature matrices
├── src/                      # Source modules
│   ├── preprocessing.py      # Signal cleaning and windowing
│   ├── features.py           # Statistical & physiological feature extraction
│   ├── loader.py             # Data loading and subject utilities
│   ├── models.py             # Machine learning classifiers definition
│   ├── evaluation.py         # Evaluation and validation metrics
│   └── __init__.py
├── scripts/                  # Executable pipeline scripts
│   ├── build_dataset.py      # Preprocesses signals and extracts features
│   └── train_evaluate.py     # Trains models and evaluates performance
├── notebooks/                # Jupyter Notebooks for exploratory data analysis (EDA)
├── experiments/              # Trained models, configurations, and evaluation reports
├── report/                   # Project documentation and academic reports
├── config.yaml               # Centralized pipeline configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** installed. It is recommended to use a virtual environment.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies include: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `pyyaml`, `scipy`, and `matplotlib`)*

### 3. Download the WESAD Dataset
1.  Download the official WESAD dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/WESAD+(Wrist+and+Chest+Affect+Detection)).
2.  Extract the zip file and place the raw subject folders (`S2`, `S3`, ..., `S17`) inside the `data/01_raw/` directory.

### 4. Run the Pipeline
The pipeline can be executed via the main scripts or by exploring the notebooks:
```bash
# Preprocess raw signals and extract features into a processed dataset
python scripts/build_dataset.py

# Train models and evaluate performance using LOSO cross-validation
python scripts/train_evaluate.py
```
