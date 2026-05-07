# Implemented Project Pipeline

This file describes the project from the code as it is implemented now. The code is the source of truth.

## What The Pipeline Does

The project implements Project 4: stress prediction from physiological signals using WESAD wrist data only.

The implemented workflow is:

```text
raw WESAD subject pickles
-> wrist-only loading
-> signal alignment and windowing
-> mixed-label transition filtering
-> handcrafted feature extraction
-> processed feature dataset
-> LOSO model training
-> multi-class evaluation
-> true binary stress evaluation
-> per-subject analysis
-> ROC/AUC, confusion matrices, reports, and report-ready figures
```

## Data Constraints

The loader reads only:

- `ACC`
- `BVP`
- `EDA`
- `TEMP`

These are taken from `payload["signal"]["wrist"]`. Chest/RespiBAN signals are not used.

The processed dataset currently has:

- 121,853 windows
- 51 features
- 15 subjects
- 0 NaN values
- 0 infinite values

Class distribution:

- Baseline: 66,859
- Stress: 36,279
- Amusement: 18,715

## Windowing

The saved window policy is:

- ACC window: 5 seconds
- BVP/EDA/TEMP window: 60 seconds
- shift: 0.25 seconds
- pure physiological windows required: yes

Mixed-label physiological windows are removed by default to reduce transition leakage.

## Models

The final code intentionally supports only a small compliant comparison:

- Logistic Regression
- Linear SVM
- PyTorch MLP
- balanced PyTorch MLP

Older tuning branches, LSTM code, and cuML-specific paths were removed to keep
the project aligned with the final reported experiment and avoid accidental
extra runs.

The prohibited original-paper models are not used in `src/` or `scripts`: Decision Tree, Random Forest, AdaBoost, LDA, and kNN.

## Validation

The code uses Leave-One-Subject-Out cross-validation.

For each fold:

1. one subject is held out;
2. all other subjects are used for training;
3. scaling and feature selection are fitted only on training subjects;
4. the held-out subject is transformed and predicted.

This avoids subject leakage.

## Evaluation Outputs

The training script saves:

- metrics CSV/JSON
- classification reports
- confusion matrices
- normalized confusion matrices
- per-subject metrics
- subject error analysis
- feature selection summary
- ROC/AUC summary files
- ROC curve figures
- optional saved LOSO predictions when `--save-loso-arrays` is used

ROC/AUC is implemented in code through `save_roc_curves`.

Current generated AUC summaries:

- binary Torch MLP stress AUC: 0.886
- binary Torch MLP Balanced stress AUC: 0.881
- multiclass Torch MLP macro AUC: 0.786
- multiclass Torch MLP Balanced macro AUC: 0.787

## Explainability

The final code keeps interpretable feature selection summaries, confusion
matrices, ROC/AUC, and per-subject error analysis. Cache-heavy SHAP and
permutation-importance helpers were removed because they were not part of the
final GPU run.

## What To Claim

It is correct to say:

- the project uses only WESAD wrist data;
- chest data is excluded;
- LOSO validation is implemented;
- preprocessing is fold-local;
- binary stress detection is the main task;
- multi-class classification is secondary;
- balanced accuracy, macro F1, confusion matrices, ROC/AUC, and per-subject errors are computed;
- the method is academic and not clinically deployable.

It is not correct to claim:

- clinical diagnosis;
- real-world deployment readiness;
- external validation;
- causal physiological biomarkers;
- end-to-end deep learning from raw signals.
