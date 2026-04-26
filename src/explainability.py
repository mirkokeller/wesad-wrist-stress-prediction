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
        n_jobs=-1,
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
    plt.close(fig)


def plot_shap_summary(
    shap_values: Any,
    X: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    class_names: list[str] | None = None,
    max_display: int = 20,
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
        plt.close("all")

        # Fallback: save per-class summary to separate file
        for i, (sv, name) in enumerate(zip(shap_values, class_names)):
            class_path = path.with_stem(f"{path.stem}_{name.lower().replace(' ', '_')}")
            shap.summary_plot(
                sv, X, feature_names=feature_names,
                max_display=max_display, show=False,
            )
            plt.tight_layout()
            plt.savefig(str(class_path), dpi=150, bbox_inches="tight")
            plt.close()
    else:
        shap.summary_plot(
            shap_values, X, feature_names=feature_names,
            max_display=max_display, show=False,
        )
        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()


def plot_shap_bar(
    shap_values: Any,
    X: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    max_display: int = 20,
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
    return shap_values, X_small
