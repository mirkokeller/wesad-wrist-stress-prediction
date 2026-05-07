"""Minimal allowed model set for the final WESAD wrist experiment."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC

try:
    _tqdm = importlib.import_module("tqdm.auto")
    tqdm = getattr(_tqdm, "tqdm")
except Exception:  # pragma: no cover
    def tqdm(iterable, **_: Any):
        return iterable

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    _torch_data = importlib.import_module("torch.utils.data")
    TensorDataset = getattr(_torch_data, "TensorDataset")
    DataLoader = getattr(_torch_data, "DataLoader")
    _HAS_TORCH = True
    _TORCH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _HAS_TORCH = False
    _TORCH_IMPORT_ERROR = str(exc)


def get_torch_status() -> dict[str, Any]:
    """Return PyTorch/CUDA availability."""
    cuda = bool(_HAS_TORCH and torch.cuda.is_available())
    return {
        "torch_available": _HAS_TORCH,
        "cuda_available": cuda,
        "device": "cuda" if cuda else "cpu",
        "device_name": torch.cuda.get_device_name(0) if cuda else "",
        "torch_import_error": _TORCH_IMPORT_ERROR,
    }


def require_cuda() -> None:
    """Fail early if the final experiment cannot run on GPU."""
    status = get_torch_status()
    if not status["torch_available"]:
        raise ImportError(f"PyTorch is unavailable: {status['torch_import_error']}")
    if not status["cuda_available"]:
        raise RuntimeError("CUDA is not available. The final run is configured to require GPU.")


def _resolve_device(device: str = "cuda") -> str:
    if not _HAS_TORCH:
        raise ImportError(f"PyTorch is unavailable: {_TORCH_IMPORT_ERROR}")

    device = device.strip().lower()
    if device not in {"cuda", "cpu", "auto"}:
        raise ValueError("device must be one of: cuda, cpu, auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible tabular MLP backed by PyTorch."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 96),
        dropout: float = 0.2,
        epochs: int = 35,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weight: str | None = None,
        device: str = "cuda",
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.device = device
        self.random_state = random_state

    def _build_model(self, input_size: int, n_classes: int):
        layers: list[Any] = []
        prev_size = input_size
        for hidden_size in self.hidden_layer_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(float(self.dropout)),
                ]
            )
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray):
        device = _resolve_device(self.device)
        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y)
        self.classes_ = np.asarray(sorted(np.unique(y_np).tolist()))
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_.tolist())}
        y_idx = np.asarray([label_to_idx[label] for label in y_np], dtype=np.int64)

        generator = torch.Generator()
        generator.manual_seed(int(self.random_state))
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_np, dtype=torch.float32),
                torch.tensor(y_idx, dtype=torch.long),
            ),
            batch_size=int(self.batch_size),
            shuffle=True,
            generator=generator,
            pin_memory=(device == "cuda"),
        )

        self.model_ = self._build_model(X_np.shape[1], len(self.classes_)).to(device)
        weight_tensor = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(np.float32)
            weights = counts.sum() / (len(self.classes_) * np.maximum(counts, 1.0))
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        self.model_.train()
        for _ in range(int(self.epochs)):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

        self.device_ = device
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_np = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_np, dtype=torch.float32)),
            batch_size=int(self.batch_size),
            shuffle=False,
            pin_memory=(self.device_ == "cuda"),
        )

        probs: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                logits = self.model_(xb.to(self.device_, non_blocking=True))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def final_models(
    random_state: int = 42,
    device: str = "cuda",
    epochs: int = 35,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Return the minimal compliant model set for the final comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            C=2.0,
            max_iter=1000,
            random_state=random_state,
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=random_state,
        ),
        "Torch MLP": TorchMLPClassifier(
            epochs=epochs,
            batch_size=batch_size,
            class_weight=None,
            device=device,
            random_state=random_state,
        ),
        "Torch MLP Balanced": TorchMLPClassifier(
            epochs=epochs,
            batch_size=batch_size,
            class_weight="balanced",
            device=device,
            random_state=random_state,
        ),
    }


def _apply_feature_pipeline_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    scaler: str = "robust",
    selection_k: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit preprocessing on train subjects only, then transform the held-out subject."""
    selected_idx = np.arange(X_train.shape[1])
    scaler_obj = StandardScaler() if scaler == "standard" else RobustScaler()
    X_train_t = scaler_obj.fit_transform(X_train)
    X_test_t = scaler_obj.transform(X_test)

    variance = VarianceThreshold(threshold=1e-12)
    X_train_t = variance.fit_transform(X_train_t)
    X_test_t = variance.transform(X_test_t)
    selected_idx = selected_idx[variance.get_support(indices=True)]

    if 0 < selection_k < X_train_t.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=int(selection_k))
        X_train_t = selector.fit_transform(X_train_t, y_train)
        X_test_t = selector.transform(X_test_t)
        selected_idx = selected_idx[selector.get_support(indices=True)]

    return X_train_t, X_test_t, selected_idx


def run_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    models: dict[str, Any],
    scaler: str = "robust",
    selection_k: int = 24,
    show_progress: bool = True,
) -> dict[str, dict[str, list[Any]]]:
    """Run leave-one-subject-out evaluation."""
    results = {
        name: {"y_true": [], "y_pred": [], "y_prob": [], "subject": [], "selected_features": []}
        for name in models
    }

    subjects = sorted(np.unique(subject).tolist())
    outer = tqdm(subjects, desc="LOSO folds", unit="subject") if show_progress else subjects
    for test_subj in outer:
        train_idx = subject != test_subj
        test_idx = subject == test_subj

        X_train, X_test, selected_idx = _apply_feature_pipeline_fold(
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_test=X[test_idx],
            scaler=scaler,
            selection_k=selection_k,
        )
        y_train = y[train_idx]
        y_test = y[test_idx]

        for name, model in models.items():
            clf = clone(model)
            clf.fit(X_train, y_train)
            y_pred = np.asarray(clf.predict(X_test)).ravel()

            results[name]["y_true"].extend(y_test.tolist())
            results[name]["y_pred"].extend(y_pred.tolist())
            if hasattr(clf, "predict_proba"):
                results[name]["y_prob"].append(np.asarray(clf.predict_proba(X_test)))
            results[name]["subject"].extend([int(test_subj)] * len(y_test))
            results[name]["selected_features"].append(selected_idx.tolist())

    return results


def compute_metrics_table(
    results: dict[str, dict[str, list[Any]]],
    sort_by: str = "Macro F1",
) -> pd.DataFrame:
    """Compute project metrics with macro F1 as the headline score."""
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        y_true = np.asarray(result["y_true"])
        y_pred = np.asarray(result["y_pred"])
        if y_true.size == 0:
            continue

        rows.append(
            {
                "Model": name,
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "Macro Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "Macro Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        )

    metrics = pd.DataFrame(rows)
    if not metrics.empty and sort_by in metrics.columns:
        metrics = metrics.sort_values(sort_by, ascending=False)
    return metrics.reset_index(drop=True)


def best_model_name(metrics_df: pd.DataFrame, metric: str = "Macro F1") -> str:
    """Return the best model according to the chosen metric."""
    if metrics_df.empty:
        raise ValueError("Cannot choose best model from an empty metrics table.")
    metric = metric if metric in metrics_df.columns else "Weighted F1"
    return str(metrics_df.loc[metrics_df[metric].astype(float).idxmax(), "Model"])


def summarize_selected_features(
    results: dict[str, dict[str, list[Any]]],
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Count how often each original feature survives fold-local selection."""
    selections = next(
        (result["selected_features"] for result in results.values() if result.get("selected_features")),
        [],
    )
    if not selections:
        return pd.DataFrame(columns=["feature_index", "feature_name", "fold_count", "selection_rate"])

    hits = np.zeros(len(feature_names), dtype=int)
    for fold_selection in selections:
        hits[np.asarray(fold_selection, dtype=int)] += 1

    selected = np.where(hits > 0)[0]
    return (
        pd.DataFrame(
            {
                "feature_index": selected,
                "feature_name": [feature_names[i] for i in selected],
                "fold_count": hits[selected],
                "selection_rate": hits[selected] / max(len(selections), 1),
            }
        )
        .sort_values(["fold_count", "feature_index"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def save_loso_results(
    results: dict[str, dict[str, list[Any]]],
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
) -> None:
    """Save predictions and probabilities only when explicitly requested."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(results)
    np.savez_compressed(
        path,
        model_names=np.array(model_names, dtype=object),
        y_true=np.array([results[name]["y_true"] for name in model_names], dtype=object),
        y_pred=np.array([results[name]["y_pred"] for name in model_names], dtype=object),
        subject=np.array([results[name]["subject"] for name in model_names], dtype=object),
        y_prob=np.array([np.vstack(results[name]["y_prob"]) for name in model_names], dtype=object),
        X=X,
        y=y,
        subject_global=subject,
        feature_names=np.array(feature_names, dtype=object),
    )
