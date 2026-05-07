"""Evaluation and plotting helpers for final LOSO outputs."""

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
    roc_auc_score,
    roc_curve,
)


def save_metrics(metrics_df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics_df.to_dict(orient="records"), indent=2), encoding="utf-8")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    label_names: dict[int, str],
) -> str:
    labels = list(label_names)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[label_names[label] for label in labels],
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
    label_names: dict[int, str],
    normalize: bool = False,
) -> pd.DataFrame:
    labels = list(label_names)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        values = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    else:
        values = cm

    names = [label_names[label] for label in labels]
    df = pd.DataFrame(values, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def compute_per_subject_metrics(y_true: np.ndarray, y_pred: np.ndarray, subject: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sid in sorted(np.unique(subject).tolist()):
        mask = subject == sid
        yt = y_true[mask]
        yp = y_pred[mask]
        rows.append(
            {
                "subject": int(sid),
                "accuracy": float(accuracy_score(yt, yp)),
                "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "support": int(yt.size),
            }
        )
    return pd.DataFrame(rows)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int) -> dict[str, Any]:
    y_true_bin = (np.asarray(y_true) == positive_label).astype(int)
    y_pred_bin = (np.asarray(y_pred) == positive_label).astype(int)

    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))

    return {
        "binary_accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "binary_balanced_accuracy": float(balanced_accuracy_score(y_true_bin, y_pred_bin)),
        "binary_precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "binary_recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "binary_f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "binary_specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "binary_support_positive": tp + fn,
        "binary_support_negative": tn + fp,
    }


def compute_subject_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, subject: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sid in sorted(np.unique(subject).tolist()):
        mask = subject == sid
        yt = y_true[mask]
        yp = y_pred[mask]
        errors = yt != yp
        rows.append(
            {
                "subject": int(sid),
                "support": int(yt.size),
                "n_errors": int(errors.sum()),
                "error_rate": float(errors.mean()) if yt.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _stack_probabilities(prob_chunks: list[Any]) -> np.ndarray | None:
    arrays = [np.asarray(chunk) for chunk in prob_chunks if np.asarray(chunk).size]
    return np.vstack(arrays) if arrays else None


def save_roc_curves(
    results: dict[str, dict[str, list[Any]]],
    output_dir: str | Path,
    label_names: dict[int, str],
    prefix: str,
    positive_label: int | None = None,
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = list(label_names)
    rows: list[dict[str, Any]] = []
    curves: list[tuple[str, str, np.ndarray, np.ndarray, float]] = []

    for model_name, result in results.items():
        y_true = np.asarray(result["y_true"])
        probs = _stack_probabilities(result["y_prob"])
        if probs is None or probs.shape[0] != y_true.shape[0]:
            continue

        class_indices = [labels.index(positive_label)] if positive_label is not None else range(len(labels))
        aucs: list[float] = []
        for class_idx in class_indices:
            if class_idx >= probs.shape[1]:
                continue
            label = labels[class_idx]
            y_binary = (y_true == label).astype(int)
            if np.unique(y_binary).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_binary, probs[:, class_idx])
            auc = float(roc_auc_score(y_binary, probs[:, class_idx]))
            class_name = label_names[label]
            aucs.append(auc)
            curves.append((model_name, class_name, fpr, tpr, auc))
            rows.append({"task": prefix, "model": model_name, "class": class_name, "auc": auc})
        if len(aucs) > 1:
            rows.append({"task": prefix, "model": model_name, "class": "macro_average", "auc": float(np.mean(aucs))})

    summary = pd.DataFrame(rows, columns=["task", "model", "class", "auc"])
    summary.to_csv(out_dir / f"{prefix}_roc_auc.csv", index=False)
    if not curves:
        return summary

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, class_name, fpr, tpr, auc in curves:
        ax.plot(fpr, tpr, label=f"{model_name} / {class_name}: {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_confusion_matrix_grid(
    results: dict[str, dict[str, list[Any]]],
    output_path: str | Path,
    label_names: dict[int, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    model_names = [name for name, result in results.items() if result["y_true"]]
    labels = list(label_names)
    n_cols = min(2, len(model_names))
    n_rows = int(np.ceil(len(model_names) / max(n_cols, 1)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes, dtype=object).reshape(-1)

    for ax, model_name in zip(axes, model_names):
        result = results[model_name]
        cm = confusion_matrix(result["y_true"], result["y_pred"], labels=labels)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=[label_names[label] for label in labels],
            yticklabels=[label_names[label] for label in labels],
        )
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for ax in axes[len(model_names):]:
        ax.axis("off")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subject_error_analysis(error_df: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    errors = error_df["error_rate"].to_numpy()
    colors = ["#d95f02" if value >= 0.3 else "#1b9e77" for value in errors]
    ax.bar(error_df["subject"].astype(str), errors, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(errors.mean(), color="#377eb8", linestyle="--", linewidth=1, label=f"mean {errors.mean():.3f}")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, min(1.0, errors.max() + 0.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
