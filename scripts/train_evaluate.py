"""Final GPU training run for WESAD wrist stress prediction."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    compute_binary_metrics,
    compute_per_subject_metrics,
    compute_subject_error_analysis,
    plot_confusion_matrix_grid,
    plot_subject_error_analysis,
    save_classification_report,
    save_confusion_matrix_csv,
    save_metrics,
    save_roc_curves,
)
from src.models import (
    best_model_name,
    compute_metrics_table,
    final_models,
    get_torch_status,
    require_cuda,
    run_loso_cv,
    save_loso_results,
    summarize_selected_features,
)

MULTICLASS_LABELS = {1: "Baseline", 2: "Stress", 3: "Amusement"}
BINARY_LABELS = {0: "Non-stress", 1: "Stress"}


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def _save_per_model_outputs(
    results: dict[str, dict[str, list]],
    metrics_df: pd.DataFrame,
    output_dir: Path,
    labels: dict[int, str],
    prefix: str,
) -> str:
    best_name = best_model_name(metrics_df)
    best = results[best_name]
    best_safe_name = _safe_name(best_name)
    y_true = np.asarray(best["y_true"])
    y_pred = np.asarray(best["y_pred"])
    subj = np.asarray(best["subject"])
    save_classification_report(
        y_true,
        y_pred,
        output_dir / f"{prefix}_best_classification_report.txt",
        labels,
    )
    save_confusion_matrix_csv(
        y_true,
        y_pred,
        output_dir / f"{prefix}_best_confusion_matrix.csv",
        labels,
    )
    compute_per_subject_metrics(y_true, y_pred, subj).to_csv(
        output_dir / f"per_subject_{prefix}_best_{best_safe_name}.csv",
        index=False,
    )
    compute_subject_error_analysis(y_true, y_pred, subj).to_csv(
        output_dir / f"{prefix}_subject_error_best_{best_safe_name}.csv",
        index=False,
    )
    return best_name


def _copy_report_figures(output_dir: Path) -> None:
    image_dir = PROJECT_ROOT / "images"
    image_dir.mkdir(exist_ok=True)
    for name in [
        "confusion_matrices_binary.png",
        "confusion_matrices_multiclass.png",
        "subject_error_best_binary.png",
        "subject_error_best_multiclass.png",
    ]:
        source = output_dir / "figures" / name
        if source.exists():
            shutil.copy2(source, image_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final CUDA Torch LOSO experiment.")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data" / "03_processed" / "features.npz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "final_gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaler", choices=["robust", "standard"], default="robust")
    parser.add_argument("--selection-k", type=int, default=24)
    parser.add_argument("--torch-epochs", type=int, default=35)
    parser.add_argument("--torch-batch-size", type=int, default=4096)
    parser.add_argument("--save-loso-arrays", action="store_true")
    args = parser.parse_args()

    require_cuda()
    status = get_torch_status()
    print(f"GPU: {status['device_name']}")

    if not args.data_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {args.data_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)

    data = np.load(args.data_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subject = data["subject"]
    feature_names = [str(name) for name in data["feature_names"]]
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(subject))} subjects")

    run_config = {
        "seed": args.seed,
        "device": "cuda",
        "scaler": args.scaler,
        "selection_k": args.selection_k,
        "torch_epochs": args.torch_epochs,
        "torch_batch_size": args.torch_batch_size,
    }
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("\nMulticlass LOSO")
    models = final_models(args.seed, "cuda", args.torch_epochs, args.torch_batch_size)
    multiclass = run_loso_cv(X, y, subject, models, scaler=args.scaler, selection_k=args.selection_k)
    multiclass_metrics = compute_metrics_table(multiclass)
    multiclass_metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    save_metrics(multiclass_metrics, args.output_dir / "metrics.json")
    best_multiclass = _save_per_model_outputs(multiclass, multiclass_metrics, args.output_dir, MULTICLASS_LABELS, "multiclass")
    plot_confusion_matrix_grid(multiclass, args.output_dir / "figures" / "confusion_matrices_multiclass.png", MULTICLASS_LABELS)
    save_roc_curves(multiclass, args.output_dir, MULTICLASS_LABELS, "multiclass")

    print("\nBinary LOSO")
    y_binary = (y == 2).astype(int)
    binary_models = final_models(args.seed, "cuda", args.torch_epochs, args.torch_batch_size)
    binary = run_loso_cv(X, y_binary, subject, binary_models, scaler=args.scaler, selection_k=args.selection_k)
    binary_metrics = compute_metrics_table(binary)
    binary_extra = pd.DataFrame(
        [
            {"Model": model_name, **compute_binary_metrics(result["y_true"], result["y_pred"], positive_label=1)}
            for model_name, result in binary.items()
        ]
    )
    binary_metrics = binary_metrics.merge(binary_extra, on="Model", how="left")
    binary_metrics.to_csv(args.output_dir / "metrics_binary.csv", index=False)
    save_metrics(binary_metrics, args.output_dir / "metrics_binary.json")
    best_binary = _save_per_model_outputs(binary, binary_metrics, args.output_dir, BINARY_LABELS, "binary")
    plot_confusion_matrix_grid(binary, args.output_dir / "figures" / "confusion_matrices_binary.png", BINARY_LABELS)
    save_roc_curves(binary, args.output_dir, BINARY_LABELS, "binary", positive_label=1)

    summarize_selected_features(multiclass, feature_names).to_csv(
        args.output_dir / "feature_selection_summary.csv",
        index=False,
    )

    plot_subject_error_analysis(
        compute_subject_error_analysis(
            np.asarray(multiclass[best_multiclass]["y_true"]),
            np.asarray(multiclass[best_multiclass]["y_pred"]),
            np.asarray(multiclass[best_multiclass]["subject"]),
        ),
        args.output_dir / "figures" / "subject_error_best_multiclass.png",
    )
    plot_subject_error_analysis(
        compute_subject_error_analysis(
            np.asarray(binary[best_binary]["y_true"]),
            np.asarray(binary[best_binary]["y_pred"]),
            np.asarray(binary[best_binary]["subject"]),
        ),
        args.output_dir / "figures" / "subject_error_best_binary.png",
    )
    _copy_report_figures(args.output_dir)

    if args.save_loso_arrays:
        save_loso_results(multiclass, X, y, subject, feature_names, args.output_dir / "loso_results.npz")
        save_loso_results(binary, X, y_binary, subject, feature_names, args.output_dir / "loso_results_binary.npz")

    print("\nFinal outputs saved to:", args.output_dir)
    print("Best multiclass:", best_multiclass)
    print("Best binary:", best_binary)


if __name__ == "__main__":
    main()
