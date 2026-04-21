# Project Brief - Stress Prediction From Physiological Signals

## Context

Project: Stress Prediction from Physiological Signals.
Goal: develop machine learning models that predict stress conditions from wearable physiological signals.
Possible task setup: binary classification (stress vs non-stress) or multi-class classification (baseline, stress, amusement).

## Required Constraints

- Use only wrist data from the WESAD dataset (Empatica E4).
- Fully exclude chest signals from RespiBAN.
- Use models different from those proposed in the original paper.
- Use leave-one-subject-out (LOSO) validation.
- Use Python.

## Content Requirements

- Show a clear understanding of the data and the real-world challenges.
- Include a short state-of-the-art review and personal critical analysis.
- Discuss data access, data quality, explainability, ethics, GDPR, and AI challenges in healthcare.
- Consider whether more advanced AI methods are useful for this project.
- Cover the points from the journal club checklist.

## Journal Club Points To Cover

- Intended use and benefit: what benefit the method gives compared with standard practice.
- Data description: acquisition, sample size, demographics, source, setting, and availability.
- Task description: the selected task and the ground-truth definition.
- Data preparation: preprocessing, possible augmentation, sampling, and data split.
- High-level method description: modeling framework and difference from prior work.
- Feature extraction: feature engineering, selection, or representation learning.
- XAI: level of explainability of the method.
- Bias, fairness, and inclusion: how diversity is handled, or which population is the focus.
- Generalizability: expected external validity and possible improvements.
- Performance evaluation: chosen metrics such as accuracy, precision, recall, F1, confusion matrix, and ROC, with justification.

## Academic Deliverables

- Maximum report length: 4 pages.
- Each section of the report should state who contributed what.
- The final PDF should include the repository link.
- Deadline: Sunday 10/05 before 12 PM.

## References

- Dataset WESAD: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html
- Paper: https://dl.acm.org/doi/epdf/10.1145/3242969.3242985
