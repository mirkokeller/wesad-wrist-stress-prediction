"""Explainability utilities: SHAP, permutation importance, feature importance plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Compute permutation feature importance.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
    X, y : np.ndarray
        Test data.
    feature_names : list[str]
        Names of each column.
    n_repeats : int
        Number of shuffles per feature.
    random_state : int
    scoring : str
        Sklearn scoring metric.

    Returns
    -------
    pd.DataFrame
        Feature importances sorted by mean decrease.
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1,
    )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return df


def plot_permutation_importance(
    imp_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 8),
    show: bool = True,
) -> None:
    """Bar plot of permutation importance with error bars."""
    plot_df = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(plot_df))
    ax.barh(
        y_pos,
        plot_df["importance_mean"].values,
        xerr=plot_df["importance_std"].values,
        align="center",
        ecolor="black",
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean accuracy decrease")
    ax.set_title(f"Permutation Feature Importance (top {top_n})")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_shap_summary(
    shap_values: Any,
    X: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    class_names: list[str] | None = None,
    max_display: int = 20,
    show: bool = True,
) -> None:
    """Save SHAP summary (beeswarm) plot for each class or globally.

    Parameters
    ----------
    shap_values : list of np.ndarray or np.ndarray
        SHAP values per class (list) or single array.
    X : np.ndarray
        Feature matrix used for explanation.
    feature_names : list[str]
    output_path : str | Path
    class_names : list[str] | None
    max_display : int
    """
    import shap

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(shap_values, list):
        n_classes = len(shap_values)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        n_cols = min(3, n_classes)
        n_rows = int(np.ceil(n_classes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = np.array(axes, dtype=object).reshape(-1)

        for i, (sv, name) in enumerate(zip(shap_values, class_names)):
            shap.summary_plot(
                sv, X, feature_names=feature_names,
                max_display=max_display, show=False,
            )
            ax = plt.gca()
            ax.set_title(name)
            fig.add_subplot(axes.flat[i])
            # plt.sca(axes.flat[i]) would need custom impl; use shap.plots instead:

        # Fallback: save per-class summary to separate file
        for i, (sv, name) in enumerate(zip(shap_values, class_names)):
            class_path = path.with_stem(f"{path.stem}_{name.lower().replace(' ', '_')}")
            shap.summary_plot(
                sv, X, feature_names=feature_names,
                max_display=max_display, show=False,
            )
            plt.tight_layout()
            plt.savefig(str(class_path), dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()
    else:
        shap.summary_plot(
            shap_values, X, feature_names=feature_names,
            max_display=max_display, show=False,
        )
        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def plot_shap_bar(
    shap_values: Any,
    X: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    max_display: int = 20,
    show: bool = True,
) -> None:
    """SHAP bar plot (mean |SHAP| per feature)."""
    import shap

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values, X, feature_names=feature_names,
        max_display=max_display, show=False, plot_type="bar",
    )
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def compute_shap_values(
    model: Any,
    X_explain: np.ndarray,
    X_background: np.ndarray | None = None,
    n_background: int = 100,
    random_state: int = 42,
) -> Any:
    """Compute SHAP values using KernelExplainer (model-agnostic).

    Parameters
    ----------
    model : fitted sklearn-compatible estimator with predict_proba.
    X_explain : np.ndarray
        Samples to explain.
    X_background : np.ndarray | None
        Background distribution. Sampled from X_explain if None.
    n_background : int
    random_state : int

    Returns
    -------
    shap_values : list or np.ndarray
    """
    import shap

    rng = np.random.RandomState(random_state)
    if X_background is None:
        if len(X_explain) > n_background:
            idx = rng.choice(len(X_explain), n_background, replace=False)
            X_background = X_explain[idx]
        else:
            X_background = X_explain

    explainer = shap.KernelExplainer(model.predict_proba, X_background)

    if len(X_explain) > 200:
        idx = rng.choice(len(X_explain), 200, replace=False)
        X_small = X_explain[idx]
    else:
        X_small = X_explain

    shap_values = explainer.shap_values(X_small, nsamples=100)

    # KernelExplainer for multi-class can return list of per-sample arrays,
    # each of shape (n_features, n_classes). Convert to standard format:
    # list of n_classes arrays, each of shape (n_samples, n_features).
    if isinstance(shap_values, list) and len(shap_values) > 0:
        sample = shap_values[0]
        if isinstance(sample, np.ndarray) and sample.ndim == 2 and sample.shape[1] > 1:
            n_classes = sample.shape[1]
            stacked = np.array(shap_values)
            shap_values = [stacked[:, :, i] for i in range(n_classes)]

    return shap_values, X_small


def plot_force_plot(
    shap_values: Any,
    X_instance: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    class_index: int = 1,
    sample_idx: int = 0,
    figsize: tuple[int, int] = (12, 3),
    show: bool = True,
) -> None:
    """Save a SHAP force plot for a single prediction.

    Parameters
    ----------
    shap_values : list of np.ndarray or np.ndarray
    X_instance : np.ndarray
        Single sample (1D or 2D with 1 row).
    feature_names : list[str]
    output_path : str | Path
    class_index : int
        Which class to explain (used when shap_values is a list).
    sample_idx : int
        Which sample to explain (index into the per-class SHAP array).
    figsize : tuple[int, int]
    """
    import shap

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sv = shap_values[class_index] if isinstance(shap_values, list) else shap_values

    X_instance_2d = X_instance.reshape(1, -1) if X_instance.ndim == 1 else X_instance

    if hasattr(sv, "values"):
        sv_values = sv.values
        base_values = sv.base_values if hasattr(sv, "base_values") else 0
    else:
        sv_values = sv
        base_values = sv.base_values if hasattr(sv, "base_values") else 0

    if isinstance(sv_values, np.ndarray) and sv_values.ndim == 3:
        sv_values = sv_values[:, :, class_index]

    if isinstance(base_values, np.ndarray):
        if base_values.ndim == 2:
            base_value = base_values[sample_idx, class_index]
        elif base_values.ndim == 1:
            base_value = base_values[sample_idx]
        else:
            base_value = base_values
    else:
        base_value = base_values

    sv_sample = sv_values[sample_idx] if getattr(sv_values, "ndim", 0) > 1 else sv_values
    features = X_instance_2d[0] if getattr(sv_sample, "ndim", 0) == 1 else X_instance_2d

    exp = shap.Explanation(
        values=sv_sample,
        base_values=base_value,
        data=features,
        feature_names=feature_names,
    )
    max_display = min(15, len(feature_names))
    shap.plots.waterfall(exp, show=False, max_display=max_display)
    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_force_plot_grid(
    shap_values: Any,
    X_samples: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    class_names: list[str] | None = None,
    n_samples: int = 6,
    random_state: int = 42,
    show: bool = True,
) -> None:
    """Save individual SHAP force plots for selected samples across classes.

    Since shap.force_plot(matplotlib=True) creates its own figure each call,
    we save one file per selected sample under the given directory.
    """
    import shap

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        n_classes = len(shap_values) if isinstance(shap_values, list) else 1
        class_names = [f"Class {i}" for i in range(n_classes)]

    rng = np.random.RandomState(random_state)
    n_classes = len(shap_values) if isinstance(shap_values, list) else 1

    for cls_idx in range(n_classes):
        sv = shap_values[cls_idx] if isinstance(shap_values, list) else shap_values

        if hasattr(sv, "values"):
            sv_values = sv.values
            base_values = sv.base_values if hasattr(sv, "base_values") else 0
        else:
            sv_values = sv
            base_values = sv.base_values if hasattr(sv, "base_values") else 0

        if isinstance(sv_values, np.ndarray) and sv_values.ndim == 3:
            sv_values = sv_values[:, :, cls_idx]

        n_avail = sv_values.shape[0]
        chosen = rng.choice(n_avail, min(n_samples, n_avail), replace=False)

        for idx in chosen:
            cls_name = class_names[cls_idx].lower().replace(" ", "_")
            file_path = path / f"force_{cls_name}_sample_{idx}.png"

            X_2d = X_samples[idx].reshape(1, -1) if X_samples[idx].ndim == 1 else X_samples[idx]

            if isinstance(base_values, np.ndarray):
                if base_values.ndim == 2:
                    base_value = base_values[idx, cls_idx]
                elif base_values.ndim == 1:
                    base_value = base_values[idx]
                else:
                    base_value = base_values
            else:
                base_value = base_values

            sv_sample = sv_values[idx] if getattr(sv_values, "ndim", 0) > 1 else sv_values
            features = X_2d[0] if getattr(sv_sample, "ndim", 0) == 1 else X_2d

            exp = shap.Explanation(
                values=sv_sample,
                base_values=base_value,
                data=features,
                feature_names=feature_names,
            )
            max_display = min(15, len(feature_names))
            shap.plots.waterfall(exp, show=False, max_display=max_display)

            fig = plt.gcf()
            fig.savefig(str(file_path), dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)


def plot_subject_error_analysis(
    error_df: pd.DataFrame,
    output_path: str | Path,
    figsize: tuple[int, int] = (12, 5),
    threshold: float = 0.3,
    show: bool = True,
) -> None:
    """Bar plot of per-subject error rate, highlighting high-error subjects.

    Parameters
    ----------
    error_df : pd.DataFrame
        Output of compute_subject_error_analysis.
    output_path : str | Path
    figsize : tuple[int, int]
    threshold : float
        Threshold above which bars are coloured red.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    subjects_str = error_df["subject"].astype(str)
    error_rates = error_df["error_rate"].values
    mean_error = error_rates.mean()

    colors = ["#e74c3c" if e >= threshold else "#2ecc71" for e in error_rates]
    ax.bar(subjects_str, error_rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=mean_error, color="blue", linestyle="--", label=f"Mean error: {mean_error:.3f}")
    ax.axhline(y=threshold, color="red", linestyle=":", alpha=0.7, label=f"Threshold: {threshold}")

    for i, (_, row) in enumerate(error_df.iterrows()):
        ax.text(i, row["error_rate"] + 0.01, str(row["n_errors"]),
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Error Rate")
    ax.set_title(f"Per-Subject Error Analysis (mean={mean_error:.3f})")
    ax.set_ylim(0, min(1.0, error_rates.max() + 0.15))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix_grid(
    results: dict[str, dict[str, list[Any]]],
    output_path: str | Path,
    label_names: dict[int, str] | None = None,
    show: bool = True,
) -> None:
    """Plot confusion matrices for all models from a results dict."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if label_names is None:
        label_names = {1: "Baseline", 2: "Stress", 3: "Amusement"}

    model_names = [
        name for name, out in results.items()
        if isinstance(out, dict) and len(out.get("y_true", [])) > 0
    ]
    labels_sorted = sorted(
        set().union(*(out["y_true"] for out in results.values()))
    )

    n_models = len(model_names)
    n_cols = min(3, max(1, n_models))
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes, dtype=object).reshape(-1)

    for ax, name in zip(axes, model_names):
        y_true = np.array(results[name]["y_true"])
        y_pred = np.array(results[name]["y_pred"])
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=[label_names.get(l, str(l)) for l in labels_sorted],
            yticklabels=[label_names.get(l, str(l)) for l in labels_sorted],
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for ax in axes[n_models:]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
