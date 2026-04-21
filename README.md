# wesad-wrist-stress-prediction

Machine Learning project for stress prediction using only wrist physiological signals from the WESAD dataset (Empatica E4).

## Objective

Build and compare models to detect stress conditions from wearable signals.

- Binary classification: stress vs non-stress
- Multi-class classification: baseline, stress, amusement

## Mandatory Constraints

- Use only wrist signals from Empatica E4
- Exclude chest signals from RespiBAN
- Use models different from those proposed in the original WESAD paper
- Use leave-one-subject-out cross-validation for subject-independent evaluation

## Dataset

- WESAD: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html
- Reference paper: https://dl.acm.org/doi/epdf/10.1145/3242969.3242985

## Planned Workflow

1. Data loading and wrist-only filtering
2. Preprocessing and windowing
3. Feature extraction (time/frequency/statistical)
4. Model training (multiple classifiers)
5. LOSO validation
6. Performance comparison and analysis
