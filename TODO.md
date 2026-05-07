# WESAD Wrist Stress Prediction — Project Checklist

## Data & Preprocessing
- [x] Load raw WESAD pickle files
- [x] Extract only wrist signals (ACC, BVP, EDA, TEMP) — exclude chest
- [x] Signal filtering (lowpass, bandpass as needed)
- [x] Windowing: ACC 5s, BVP/EDA/TEMP 60s, shift 0.25s
- [x] Transition-safe windowing: discard mixed-label 60s physiological windows by default
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
- [x] Class-balanced SVM and Logistic Regression variants for imbalanced labels
- [x] Logistic Regression (baseline + tuned: C=2.0)
- [x] MLP (200, 80) with early stopping
- [x] CUDA-backed Torch MLP and Torch MLP Balanced
- [x] LSTM (2-layer, 96 hidden, seq_len=8)
- [x] GPU backend support (cuML + CUDA)
- [x] LOSO cross-validation

## Evaluation
- [x] Accuracy, Precision, Recall, F1-score (weighted) table
- [x] Macro-F1 and balanced accuracy for imbalanced-class reporting
- [x] Confusion matrices per model
- [x] ROC curves and AUC summaries (One-vs-Rest, saved by `save_roc_curves`)
- [x] Metrics JSON persistence helper (`save_metrics_json`)
- [x] Per-subject performance breakdown (`compute_per_subject_metrics`)
- [x] True binary classification (stress vs. non-stress) LOSO training and evaluation
- [x] Collapsed binary-from-multiclass metrics saved only as diagnostic
- [x] Metrics bar plot (grouped bar chart saved to figures/)
- [x] Per-subject accuracy bar plot (best model, saved to figures/)

## Explainability (XAI)
- [x] SHAP summary plots (`src/explainability.py` — `compute_shap_values`, `plot_shap_summary`, `plot_shap_bar`)
- [x] SHAP force plots (per-sample explanations) (`plot_force_plot`, `plot_force_plot_grid`)
- [x] Permutation feature importance (`src/explainability.py` — `compute_permutation_importance`, `plot_permutation_importance`)
- [x] Held-out LOSO permutation importance for best model, avoiding all-data XAI leakage
- [x] Per-class feature importance analysis (SHAP per-class summary plots)
- [x] Error analysis: which subjects/samples are misclassified most (`compute_subject_error_analysis`, `plot_subject_error_analysis`)

## CLI & Scripts
- [x] `scripts/build_dataset.py` — raw → features
- [x] `scripts/train_evaluate.py` — run full training + evaluation + XAI

- [x] Best-model classification report selected by Macro-F1 instead of model insertion order
- [x] Saved LOSO probabilities can be reloaded for ROC/AUC analysis

## Analysis Notebooks
- [x] `01-eda.ipynb` — raw data exploration
- [x] `02-preprocessing.ipynb` — preprocessing & windowing validation
- [x] `03-features.ipynb` — extracted feature analysis
- [x] `04-models.ipynb` — model training and metrics
- [x] `05-explainability.ipynb` — XAI, feature importance, error analysis

## Report Requirements
- [ ] Journal Club checklist coverage
- [ ] State-of-the-art analysis / related work
- [ ] Data quality, ethics, GDPR discussion
- [ ] Explainability discussion
- [ ] Bias, fairness, and inclusion analysis
- [ ] Generalizability assessment
- [ ] Advanced AI techniques consideration
