"""Full training and evaluation pipeline for WESAD wrist stress prediction.

Usage:
    python scripts/train_evaluate.py
    python scripts/train_evaluate.py --data-path data/03_processed/features.npz
    python scripts/train_evaluate.py --backend cpu --skip-xai
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    get_backend_status,
    get_enhanced_models,
    get_torch_status,
    run_loso_cv,
    compute_metrics_table,
    summarize_selected_features,
    save_loso_results,
)
from src.evaluation import (
    compute_per_subject_metrics,
    compute_binary_metrics,
    compute_binary_per_subject,
    compute_subject_error_analysis,
    save_metrics,
    save_classification_report,
    save_confusion_matrix_csv,
    save_roc_curves,
)
from src.explainability import (
    compute_loso_permutation_importance,
    plot_permutation_importance,
    plot_confusion_matrix_grid,
    plot_subject_error_analysis,
)


MULTICLASS_LABEL_NAMES = {1: "Baseline", 2: "Stress", 3: "Amusement"}
BINARY_LABEL_NAMES = {0: "Non-stress", 1: "Stress"}


def _is_torch_model(model: object) -> bool:
    """Return True for project estimators that actually train with PyTorch."""
    return model.__class__.__name__ == "TorchMLPClassifier"


def _best_model_name(metrics_df: pd.DataFrame, primary_metric: str = "Macro F1") -> str | None:
    """Return the model name with the strongest validation metric."""
    if metrics_df.empty:
        return None
    metric = primary_metric if primary_metric in metrics_df.columns else "F1-Score"
    best_idx = metrics_df[metric].astype(float).idxmax()
    return str(metrics_df.loc[best_idx, "Model"])


def _save_model_diagnostics(
    results: dict[str, dict[str, list]],
    metrics_df: pd.DataFrame,
    output_dir: Path,
    label_names: dict[int, str],
    prefix: str,
) -> str | None:
    """Save per-model reports, confusion matrices, and error tables."""
    reports_dir = output_dir / "reports"
    cms_dir = output_dir / "confusion_matrices"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cms_dir.mkdir(parents=True, exist_ok=True)

    for model_name, result in results.items():
        yt = np.array(result["y_true"])
        yp = np.array(result["y_pred"])
        if len(yt) == 0:
            continue

        safe_name = model_name.lower().replace(" ", "_")
        save_classification_report(
            yt,
            yp,
            reports_dir / f"{prefix}_{safe_name}_classification_report.txt",
            label_names=label_names,
        )
        save_confusion_matrix_csv(
            yt,
            yp,
            cms_dir / f"{prefix}_{safe_name}_confusion_matrix.csv",
            label_names=label_names,
        )
        save_confusion_matrix_csv(
            yt,
            yp,
            cms_dir / f"{prefix}_{safe_name}_confusion_matrix_normalized.csv",
            label_names=label_names,
            normalize=True,
        )

        if result.get("subject"):
            error_df = compute_subject_error_analysis(yt, yp, np.array(result["subject"]))
            error_df.to_csv(
                output_dir / f"{prefix}_subject_error_{safe_name}.csv",
                index=False,
            )

    best_model = _best_model_name(metrics_df)
    if best_model and best_model in results:
        best_result = results[best_model]
        report = save_classification_report(
            np.array(best_result["y_true"]),
            np.array(best_result["y_pred"]),
            output_dir / f"{prefix}_best_classification_report.txt",
            label_names=label_names,
        )
        print(f"\nBest {prefix} model: {best_model}\n{report}")
    return best_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate models on WESAD wrist features."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "03_processed" / "features.npz",
        help="Path to the processed features .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "run_001",
        help="Output directory for metrics, figures, and reports.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Backend for model training ('auto', 'cpu', 'gpu').",
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip explainability analysis (faster).",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="robust",
        choices=["robust", "standard"],
        help="Fold-wise scaler fitted only on training subjects.",
    )
    parser.add_argument(
        "--selection-k",
        type=int,
        default=24,
        help="Number of fold-wise selected features. Use 0 to disable selection.",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="f_classif",
        choices=["f_classif", "mutual_info"],
        help="Fold-wise univariate feature selection method.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Optional comma-separated model names to run. "
            "Example: 'Logistic Regression,Logistic Regression Balanced,MLP'."
        ),
    )
    parser.add_argument(
        "--torch-epochs",
        type=int,
        default=None,
        help="Override epochs for Torch MLP models.",
    )
    parser.add_argument(
        "--torch-batch-size",
        type=int,
        default=None,
        help="Override batch size for Torch MLP models.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Processed feature file not found: {args.data_path}. "
            "Place WESAD under data/01_raw/WESAD and run scripts/build_dataset.py first."
        )
    data = np.load(str(args.data_path), allow_pickle=True)
    X, y = data["X"], data["y"]
    subject = data["subject"]
    feature_names = list(data["feature_names"])
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Subjects: {len(np.unique(subject))}")

    # ── Configure models ───────────────────────────────────────────────────
    backend_status = get_backend_status()
    torch_status = get_torch_status()
    print(f"  Backend: {args.backend} (cuML: {backend_status['cuml_available']})")
    print(f"  PyTorch: {torch_status['torch_available']}, CUDA: {torch_status['cuda_available']}")

    models = get_enhanced_models(random_state=args.seed, backend=args.backend)
    if args.models.strip():
        requested = [name.strip() for name in args.models.split(",") if name.strip()]
        missing = [name for name in requested if name not in models]
        if missing:
            raise ValueError(f"Unknown model(s): {missing}. Available: {list(models)}")
        models = {name: models[name] for name in requested}
    elif args.backend == "gpu" and not backend_status["cuml_available"]:
        models = {name: model for name, model in models.items() if _is_torch_model(model)}

    if args.backend == "gpu" and not backend_status["cuml_available"]:
        cpu_only = [name for name, model in models.items() if not _is_torch_model(model)]
        if cpu_only:
            raise RuntimeError(
                "GPU backend requested, but cuML is not available for sklearn-style "
                f"models: {cpu_only}. Use Torch models or install cuML/RAPIDS."
            )
    for model in models.values():
        if _is_torch_model(model):
            if args.torch_epochs is not None:
                model.epochs = args.torch_epochs
            if args.torch_batch_size is not None:
                model.batch_size = args.torch_batch_size
    print(f"  Models: {list(models.keys())}")

    run_lstm = not args.skip_lstm and torch_status["torch_available"]
    if args.skip_lstm:
        print("  LSTM: skipped (--skip-lstm)")
    elif not torch_status["torch_available"]:
        print("  LSTM: skipped (PyTorch unavailable)")
    else:
        print("  LSTM: enabled")

    # Fold-local feature pipeline. Parameters are explicit CLI choices.
    best_config = {
        "scaler": args.scaler,
        "selection_method": args.selection_method,
        "variance_threshold": 1e-12,
    }
    if args.selection_k > 0:
        best_config["selection_k"] = args.selection_k
    print(f"  Feature pipeline: {best_config}")

    lstm_config = {
        "sequence_length": 8,
        "stride": 1,
        "hidden_size": 96,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 8,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "device": "auto",
        "random_state": args.seed,
    }

    # ── Run LOSO ───────────────────────────────────────────────────────────
    print("\nRunning LOSO cross-validation...")
    start = time.time()

    results = run_loso_cv(
        X=X,
        y=y,
        subject=subject,
        models=models,
        show_progress=True,
        backend=args.backend,
        feature_pipeline_config=best_config,
        include_lstm=run_lstm,
        lstm_config=lstm_config,
    )

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")

    # ── Persist LOSO results for later XAI notebook ─────────────────────────
    loso_results_path = output_dir / "loso_results"
    save_loso_results(results, X, y, subject, feature_names, loso_results_path)
    print(f"  Saved LOSO results to: {loso_results_path}.npz")

    # ── Metrics: multi-class ────────────────────────────────────────────────
    print("\nComputing multi-class metrics...")
    metrics_df = compute_metrics_table(results)
    print(metrics_df.to_string(index=False))

    save_metrics(metrics_df, output_dir / "metrics.json")
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    best_multiclass_model = _save_model_diagnostics(
        results=results,
        metrics_df=metrics_df,
        output_dir=output_dir,
        label_names=MULTICLASS_LABEL_NAMES,
        prefix="multiclass",
    )
    if best_multiclass_model and best_multiclass_model in results:
        save_classification_report(
            np.array(results[best_multiclass_model]["y_true"]),
            np.array(results[best_multiclass_model]["y_pred"]),
            output_dir / "classification_report.txt",
            label_names=MULTICLASS_LABEL_NAMES,
        )
    plot_confusion_matrix_grid(
        results,
        output_dir / "figures" / "confusion_matrices_multiclass.png",
        label_names=MULTICLASS_LABEL_NAMES,
        show=False,
    )
    multiclass_roc_df = save_roc_curves(
        results,
        output_dir,
        label_names=MULTICLASS_LABEL_NAMES,
        prefix="multiclass",
    )
    if not multiclass_roc_df.empty:
        print("\nMulti-class ROC AUC:")
        print(multiclass_roc_df.to_string(index=False))

    # ── Metrics: per-subject ────────────────────────────────────────────────
    print("\nComputing per-subject metrics...")
    for model_name, result in results.items():
        yt = np.array(result["y_true"])
        yp = np.array(result["y_pred"])
        subj = np.array(result["subject"])
        ps_df = compute_per_subject_metrics(yt, yp, subj)
        ps_df.to_csv(output_dir / f"per_subject_{model_name.lower().replace(' ', '_')}.csv", index=False)
        print(f"  {model_name}: per-subject accuracy range [{ps_df['accuracy'].min():.3f}, {ps_df['accuracy'].max():.3f}]")

    # ── Metrics: binary ────────────────────────────────────────────────────
    print("\nComputing collapsed binary metrics from multi-class predictions (diagnostic)...")
    collapsed_binary_records = []
    for model_name, result in results.items():
        yt = np.array(result["y_true"])
        yp = np.array(result["y_pred"])
        bm = compute_binary_metrics(yt, yp, positive_label=2)
        bm["model"] = model_name
        bm["source"] = "collapsed_from_multiclass"
        collapsed_binary_records.append(bm)

    collapsed_binary_df = pd.DataFrame(collapsed_binary_records)
    collapsed_binary_df.to_csv(output_dir / "metrics_binary_from_multiclass.csv", index=False)

    print("\nRunning true binary LOSO cross-validation (stress vs. non-stress)...")
    y_binary = (y == 2).astype(int)
    binary_models = get_enhanced_models(random_state=args.seed, backend=args.backend)
    if args.models.strip():
        binary_models = {name: binary_models[name] for name in models}
    elif args.backend == "gpu" and not backend_status["cuml_available"]:
        binary_models = {
            name: model for name, model in binary_models.items() if _is_torch_model(model)
        }
    for model in binary_models.values():
        if _is_torch_model(model):
            if args.torch_epochs is not None:
                model.epochs = args.torch_epochs
            if args.torch_batch_size is not None:
                model.batch_size = args.torch_batch_size
    binary_results = run_loso_cv(
        X=X,
        y=y_binary,
        subject=subject,
        models=binary_models,
        show_progress=True,
        backend=args.backend,
        feature_pipeline_config=best_config,
        include_lstm=run_lstm,
        lstm_config=lstm_config,
    )

    binary_loso_results_path = output_dir / "loso_results_binary"
    save_loso_results(binary_results, X, y_binary, subject, feature_names, binary_loso_results_path)

    print("\nComputing true binary metrics...")
    binary_metrics_df = compute_metrics_table(binary_results)
    binary_extra_records = []
    for model_name, result in binary_results.items():
        bm = compute_binary_metrics(
            np.array(result["y_true"]),
            np.array(result["y_pred"]),
            positive_label=1,
        )
        bm["Model"] = model_name
        binary_extra_records.append(bm)

    binary_extra_df = pd.DataFrame(binary_extra_records)
    binary_metrics_df = binary_metrics_df.merge(binary_extra_df, on="Model", how="left")
    print(binary_metrics_df.to_string(index=False))
    save_metrics(binary_metrics_df, output_dir / "metrics_binary.json")
    binary_metrics_df.to_csv(output_dir / "metrics_binary.csv", index=False)

    best_binary_model = _save_model_diagnostics(
        results=binary_results,
        metrics_df=binary_metrics_df,
        output_dir=output_dir,
        label_names=BINARY_LABEL_NAMES,
        prefix="binary",
    )
    plot_confusion_matrix_grid(
        binary_results,
        output_dir / "figures" / "confusion_matrices_binary.png",
        label_names=BINARY_LABEL_NAMES,
        show=False,
    )
    binary_roc_df = save_roc_curves(
        binary_results,
        output_dir,
        label_names=BINARY_LABEL_NAMES,
        prefix="binary",
        positive_label=1,
    )
    if not binary_roc_df.empty:
        print("\nBinary ROC AUC:")
        print(binary_roc_df.to_string(index=False))

    for model_name, result in binary_results.items():
        yt = np.array(result["y_true"])
        yp = np.array(result["y_pred"])
        subj = np.array(result["subject"])
        ps_bin = compute_binary_per_subject(yt, yp, subj, positive_label=1)
        ps_bin.to_csv(
            output_dir / f"per_subject_binary_{model_name.lower().replace(' ', '_')}.csv",
            index=False,
        )

    # ── Feature summary ─────────────────────────────────────────────────────
    print("\nTop selected features:")
    feat_summary = summarize_selected_features(results, feature_names, top_n=20)
    print(feat_summary.to_string(index=False))
    feat_summary.to_csv(output_dir / "feature_selection_summary.csv", index=False)

    # ── Explainability (permutation importance) ────────────────────────────
    if not args.skip_xai:
        print("\nRunning held-out LOSO permutation importance (XAI)...")
        xai_jobs = [
            ("multiclass", best_multiclass_model, models, y),
            ("binary", best_binary_model, binary_models, y_binary),
        ]
        for task_name, model_name, model_registry, task_y in xai_jobs:
            if not model_name or model_name == "LSTM":
                print(f"  {task_name}: skipped fold-wise permutation importance for {model_name}")
                continue

            model = model_registry[model_name]
            imp_df = compute_loso_permutation_importance(
                X=X,
                y=task_y,
                subject=subject,
                model=model,
                feature_names=feature_names,
                feature_pipeline_config=best_config,
                n_repeats=2,
                random_state=args.seed,
                scoring="balanced_accuracy",
            )
            safe_name = model_name.lower().replace(" ", "_")
            imp_df.to_csv(
                output_dir / f"perm_importance_loso_{task_name}_{safe_name}.csv",
                index=False,
            )
            plot_permutation_importance(
                imp_df,
                output_dir / "figures" / f"perm_importance_loso_{task_name}_{safe_name}.png",
                top_n=15,
                show=False,
            )
            if not imp_df.empty:
                print(
                    f"  {task_name}/{model_name}: top held-out feature = "
                    f"{imp_df.iloc[0]['feature']} ({imp_df.iloc[0]['importance_mean']:.4f})"
                )

    if best_multiclass_model and best_multiclass_model in results:
        error_df = compute_subject_error_analysis(
            np.array(results[best_multiclass_model]["y_true"]),
            np.array(results[best_multiclass_model]["y_pred"]),
            np.array(results[best_multiclass_model]["subject"]),
        )
        plot_subject_error_analysis(
            error_df,
            output_dir / "figures" / "subject_error_best_multiclass.png",
            show=False,
        )

    if best_binary_model and best_binary_model in binary_results:
        error_df = compute_subject_error_analysis(
            np.array(binary_results[best_binary_model]["y_true"]),
            np.array(binary_results[best_binary_model]["y_pred"]),
            np.array(binary_results[best_binary_model]["subject"]),
        )
        plot_subject_error_analysis(
            error_df,
            output_dir / "figures" / "subject_error_best_binary.png",
            show=False,
        )

    print(f"\nDone. All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
