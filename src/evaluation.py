"""Evaluation utilities for LOSO: per-subject metrics, binary classification, persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    label_names: dict[int, str] | None = None,
) -> str:
    """Save sklearn classification_report to a text file and return it."""
    report = classification_report(
        y_true, y_pred, target_names=list(label_names.values()) if label_names else None,
        zero_division=0,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report
