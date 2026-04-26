# WESAD Wrist Stress Prediction — Project Checklist

## Data & Preprocessing
- [x] Load raw WESAD pickle files
- [x] Extract only wrist signals (ACC, BVP, EDA, TEMP) — exclude chest
- [x] Signal filtering (lowpass, bandpass as needed)
- [x] Windowing: ACC 5s, BVP/EDA/TEMP 60s, shift 0.25s
- [x] Label downsampling (700 Hz → 4 Hz via mode)
- [x] Feature extraction pipeline (48 → 51 features)
- [x] Build dataset script (`scripts/build_dataset.py`)

## Feature Engineering
- [x] ACC: per-axis stats + magnitude stats + peak frequencies (24)
- [x] BVP: HR + HRV time-domain + HRV frequency-domain + SampEn (12)
- [x] EDA: raw stats + SCL/SCR decomposition + SCR count (10)
- [x] TEMP: stats (5)
- [x] Feature selection pipeline (f_classif, mutual_info, variance/correlation pruning, PCA)

## Models
- [x] SVM (baseline + tuned: C=6.0, gamma=scale)
- [x] Logistic Regression (baseline + tuned: C=2.0)
- [x] MLP (200, 80) with early stopping
- [x] LSTM (2-layer, 96 hidden, seq_len=8)
- [x] GPU backend support (cuML + CUDA)
- [x] LOSO cross-validation

## Evaluation
- [x] Accuracy, Precision, Recall, F1-score (weighted) table
- [x] Confusion matrices per model
- [x] ROC curves (One-vs-Rest) per model
- [ ] Metrics JSON persistence helper
- [ ] Per-subject performance breakdown
- [ ] Binary classification (stress vs. non-stress) evaluation
- [ ] Per-subject error analysis

## Explainability (XAI)
- [ ] SHAP summary plots (global feature importance)
- [ ] SHAP force plots (per-sample explanations)
- [ ] Permutation feature importance
- [ ] Per-class feature importance analysis
- [ ] Error analysis: which subjects/samples are misclassified most

## CLI & Scripts
- [x] `scripts/build_dataset.py` — raw → features
- [ ] `scripts/train_evaluate.py` — run full training + evaluation + XAI

## Analysis Notebooks
- [x] `01-eda.ipynb` — raw data exploration
- [x] `02-preprocessing.ipynb` — preprocessing & windowing validation
- [x] `03-features.ipynb` — extracted feature analysis
- [x] `04-models.ipynb` — model training and metrics
- [ ] `05-explainability.ipynb` — XAI, feature importance, error analysis

## Report Requirements
- [ ] Journal Club checklist coverage
- [ ] State-of-the-art analysis / related work
- [ ] Data quality, ethics, GDPR discussion
- [ ] Explainability discussion
- [ ] Bias, fairness, and inclusion analysis
- [ ] Generalizability assessment
- [ ] Advanced AI techniques consideration
