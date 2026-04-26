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
)
from src.evaluation import (
    compute_per_subject_metrics,
    compute_binary_metrics,
    compute_binary_per_subject,
    save_metrics,
    save_classification_report,
)
from src.explainability import (
    compute_permutation_importance,
    plot_permutation_importance,
)


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
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
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
    print(f"  Models: {list(models.keys())}")

    run_lstm = not args.skip_lstm and torch_status["torch_available"]
    if args.skip_lstm:
        print("  LSTM: skipped (--skip-lstm)")
    elif not torch_status["torch_available"]:
        print("  LSTM: skipped (PyTorch unavailable)")
    else:
        print("  LSTM: enabled")

    # ── Best feature config from notebook ──────────────────────────────────
    best_config = {
        "scaler": "robust",
        "selection_k": 24,
        "selection_method": "f_classif",
    }

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

    # ── Metrics: multi-class ────────────────────────────────────────────────
    print("\nComputing multi-class metrics...")
    metrics_df = compute_metrics_table(results)
    print(metrics_df.to_string(index=False))

    save_metrics(metrics_df, output_dir / "metrics.json")
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    # Save classification report for best model
    if not metrics_df.empty:
        best_model = metrics_df.iloc[0]["Model"]
        if best_model in results:
            report = save_classification_report(
                np.array(results[best_model]["y_true"]),
                np.array(results[best_model]["y_pred"]),
                output_dir / "classification_report.txt",
                label_names={1: "Baseline", 2: "Stress", 3: "Amusement"},
            )
            print(f"\nClassification report ({best_model}):\n{report}")

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
    print("\nComputing binary metrics (stress vs. non-stress)...")
    binary_records = []
    for model_name, result in results.items():
        yt = np.array(result["y_true"])
        yp = np.array(result["y_pred"])
        bm = compute_binary_metrics(yt, yp, positive_label=2)
        bm["model"] = model_name
        binary_records.append(bm)

    binary_df = pd.DataFrame(binary_records)
    binary_df = binary_df[["model", "binary_accuracy", "binary_f1", "binary_precision",
                           "binary_recall", "binary_specificity"]]
    print(binary_df.to_string(index=False))
    binary_df.to_csv(output_dir / "metrics_binary.csv", index=False)

    # Per-subject binary
    for model_name, result in results.items():
        yt = np.array(result["y_true"])
        yp = np.array(result["y_pred"])
        subj = np.array(result["subject"])
        ps_bin = compute_binary_per_subject(yt, yp, subj, positive_label=2)
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
        print("\nRunning permutation importance (XAI)...")
        for model_name, result in results.items():
            if model_name == "LSTM":
                continue  # LSTM needs sequence data
            if not result.get("y_prob"):
                continue

            yt = np.array(result["y_true"])
            subj = np.array(result["subject"])

            model = models[model_name]
            clf = model.__class__(**model.get_params())
            clf.fit(X, y)

            imp_df = compute_permutation_importance(
                clf, X, y, feature_names, n_repeats=5, random_state=args.seed,
            )
            imp_df.to_csv(
                output_dir / f"perm_importance_{model_name.lower().replace(' ', '_')}.csv",
                index=False,
            )
            plot_permutation_importance(
                imp_df,
                output_dir / "figures" / f"perm_importance_{model_name.lower().replace(' ', '_')}.png",
                top_n=15,
            )
            print(f"  {model_name}: top feature = {imp_df.iloc[0]['feature']} ({imp_df.iloc[0]['importance_mean']:.4f})")

    print(f"\nDone. All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
