"""Evaluation utilities for LOSO: per-subject metrics, binary classification, persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject: np.ndarray,
    subject_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Compute metrics per subject.

    Parameters
    ----------
    y_true : np.ndarray
        True labels for all samples.
    y_pred : np.ndarray
        Predicted labels for all samples.
    subject : np.ndarray
        Subject id for each sample.
    subject_ids : list[int] | None
        Subset of subjects to report. Uses all present if None.

    Returns
    -------
    pd.DataFrame
        Per-subject accuracy, precision, recall, F1, and support.
    """
    unique_subjects = sorted(np.unique(subject).tolist())
    if subject_ids is not None:
        unique_subjects = [s for s in unique_subjects if s in subject_ids]

    records: list[dict[str, Any]] = []
    for sid in unique_subjects:
        mask = subject == sid
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) == 0:
            continue
        records.append(
            {
                "subject": sid,
                "accuracy": float(accuracy_score(yt, yp)),
                "precision": float(
                    precision_score(yt, yp, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(yt, yp, average="weighted", zero_division=0)
                ),
                "f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
                "support": int(len(yt)),
            }
        )

    return pd.DataFrame(records)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: int = 2,
) -> dict[str, float]:
    """Compute binary metrics for a stress vs. non-stress formulation.

    Non-stress = everything not equal to positive_label.

    Parameters
    ----------
    y_true : np.ndarray
        Multi-class true labels.
    y_pred : np.ndarray
        Multi-class predicted labels.
    positive_label : int, default 2
        The label treated as "stress".

    Returns
    -------
    dict[str, float]
        Binary accuracy, precision, recall, F1, specificity.
    """
    y_true_bin = (np.asarray(y_true) == positive_label).astype(int)
    y_pred_bin = (np.asarray(y_pred) == positive_label).astype(int)

    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "binary_accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "binary_balanced_accuracy": float(balanced_accuracy_score(y_true_bin, y_pred_bin)),
        "binary_precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "binary_recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "binary_f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "binary_specificity": specificity,
        "binary_support_positive": int(tp + fn),
        "binary_support_negative": int(tn + fp),
    }


def compute_binary_per_subject(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject: np.ndarray,
    positive_label: int = 2,
) -> pd.DataFrame:
    """Compute per-subject binary metrics."""
    records: list[dict[str, Any]] = []
    for sid in sorted(np.unique(subject).tolist()):
        mask = subject == sid
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) == 0:
            continue
        bm = compute_binary_metrics(yt, yp, positive_label=positive_label)
        bm["subject"] = sid
        bm["support"] = int(len(yt))
        records.append(bm)

    return pd.DataFrame(records)


def save_metrics(metrics_df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist metrics table to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_df.to_dict(orient="records"), handle, indent=2)


def compute_subject_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject: np.ndarray,
    subject_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Analyze per-subject misclassification counts and error rate.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    subject : np.ndarray
        Subject id per sample.
    subject_ids : list[int] | None
        Subset of subjects.

    Returns
    -------
    pd.DataFrame
        Columns: subject, support, n_errors, error_rate, misclass_details.
    """
    unique_subjects = sorted(np.unique(subject).tolist())
    if subject_ids is not None:
        unique_subjects = [s for s in unique_subjects if s in subject_ids]

    records: list[dict[str, Any]] = []
    for sid in unique_subjects:
        mask = subject == sid
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) == 0:
            continue

        errors = yt != yp
        n_errors = int(errors.sum())
        error_rate = float(n_errors / len(yt))

        misclass_details: str = ""
        for true_label in sorted(set(yt.tolist())):
            mask_true = yt == true_label
            preds = yp[mask_true]
            n_wrong = int((preds != true_label).sum())
            if n_wrong > 0:
                top_wrong = sorted(
                    [(int(l), int((preds == l).sum())) for l in set(preds.tolist()) if l != true_label],
                    key=lambda x: -x[1],
                )[:3]
                details = ", ".join(f"pred→{l}:{c}" for l, c in top_wrong)
                misclass_details += f"true={true_label} [{details}]; "

        records.append(
            {
                "subject": sid,
                "support": int(len(yt)),
                "n_errors": n_errors,
                "error_rate": round(error_rate, 4),
                "misclass_details": misclass_details.strip("; "),
            }
        )

    return pd.DataFrame(records)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    label_names: dict[int, str] | None = None,
) -> str:
    """Save sklearn classification_report to a text file and return it."""
    labels = list(label_names.keys()) if label_names else None
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(label_names.values()) if label_names else None,
        zero_division=0,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report


def save_confusion_matrix_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    label_names: dict[int, str] | None = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Save a confusion matrix as CSV and return it as a dataframe."""
    if label_names:
        labels = list(label_names.keys())
        names = [label_names[label] for label in labels]
    else:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
        names = [str(label) for label in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_values = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    else:
        cm_values = cm

    df = pd.DataFrame(
        cm_values,
        index=[f"true_{name}" for name in names],
        columns=[f"pred_{name}" for name in names],
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df
