"""Model utilities for LOSO training and evaluation on WESAD wrist features."""

from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

try:
    _tqdm_module = importlib.import_module("tqdm.auto")
    tqdm = getattr(_tqdm_module, "tqdm")
except Exception:  # pragma: no cover - fallback when tqdm isn't installed
    def tqdm(iterable, **kwargs):
        return iterable

_HAS_CUML = False
_CUML_IMPORT_ERROR = ""

try:
    _cuml_linear_model = importlib.import_module("cuml.linear_model")
    _cuml_svm = importlib.import_module("cuml.svm")

    cuLogisticRegression = getattr(_cuml_linear_model, "LogisticRegression")
    cuSVC = getattr(_cuml_svm, "SVC")

    _HAS_CUML = True
except Exception as exc:  # pragma: no cover - depends on local environment
    _CUML_IMPORT_ERROR = str(exc)

_HAS_TORCH = False
_TORCH_IMPORT_ERROR = ""

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    _torch_data = importlib.import_module("torch.utils.data")
    TensorDataset = getattr(_torch_data, "TensorDataset")
    DataLoader = getattr(_torch_data, "DataLoader")
    _HAS_TORCH = True
except Exception as exc:  # pragma: no cover - depends on local environment
    _TORCH_IMPORT_ERROR = str(exc)


def resolve_backend(backend: str = "auto") -> str:
    """Resolve backend choice to either 'cpu' or 'gpu'.

    backend='auto' uses GPU when cuML is available.
    backend='gpu' uses cuML for sklearn-style estimators when available. If
    cuML is unavailable but Torch CUDA is available, sklearn-style estimators
    still run on CPU while Torch estimators use CUDA.
    """
    backend_normalized = backend.strip().lower()
    if backend_normalized not in {"auto", "cpu", "gpu"}:
        raise ValueError("backend must be one of: 'auto', 'cpu', 'gpu'")

    if backend_normalized == "cpu":
        return "cpu"

    if backend_normalized == "gpu":
        if not _HAS_CUML:
            if _HAS_TORCH and torch.cuda.is_available():
                return "cpu"
            raise ImportError(
                "GPU backend requested but cuML is unavailable. "
                "Install RAPIDS/cuML for your CUDA version. "
                f"Import error: {_CUML_IMPORT_ERROR}"
            )
        return "gpu"

    return "gpu" if _HAS_CUML else "cpu"


def get_backend_status() -> dict[str, Any]:
    """Return backend availability information for notebook logging."""
    return {
        "cuml_available": _HAS_CUML,
        "auto_backend": "gpu" if _HAS_CUML else "cpu",
        "cuml_import_error": _CUML_IMPORT_ERROR,
        "torch_cuda_available": bool(_HAS_TORCH and torch.cuda.is_available()),
    }


def get_torch_status() -> dict[str, Any]:
    """Return PyTorch availability information for LSTM training."""
    cuda_available = bool(_HAS_TORCH and torch.cuda.is_available())
    return {
        "torch_available": _HAS_TORCH,
        "cuda_available": cuda_available,
        "default_device": "cuda" if cuda_available else "cpu",
        "torch_import_error": _TORCH_IMPORT_ERROR,
    }


def get_default_models(random_state: int = 42, backend: str = "auto") -> dict[str, Any]:
    """Return baseline models allowed by the project constraints."""
    resolved_backend = resolve_backend(backend)

    if resolved_backend == "gpu":
        return {
            "SVM": cuSVC(kernel="rbf", probability=True, random_state=random_state),
            "Logistic Regression": cuLogisticRegression(max_iter=1000),
            # cuML does not provide a stable MLP equivalent for this workflow.
            "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state),
        }

    return {
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state),
    }


def get_enhanced_models(random_state: int = 42, backend: str = "auto") -> dict[str, Any]:
    """Return stronger baseline models for tabular features.

    Includes SVM (baseline + tuned), Logistic Regression (baseline + tuned),
    and MLP. Uses cuML on GPU backend, sklearn on CPU.
    """
    resolved_backend = resolve_backend(backend)

    if resolved_backend == "gpu":
        models: dict[str, Any] = {
            "SVM": cuSVC(kernel="rbf", probability=True, random_state=random_state),
            "SVM Tuned": cuSVC(C=6.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state),
            "Logistic Regression": cuLogisticRegression(max_iter=1000),
            "Logistic Regression Tuned": cuLogisticRegression(C=2.0, max_iter=1000),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(200, 80),
                max_iter=700,
                early_stopping=True,
                random_state=random_state,
            ),
        }
        if _HAS_TORCH:
            models["Torch MLP"] = TorchMLPClassifier(
                hidden_layer_sizes=(256, 96),
                epochs=35,
                batch_size=4096,
                class_weight=None,
                device="auto",
                random_state=random_state,
            )
            models["Torch MLP Balanced"] = TorchMLPClassifier(
                hidden_layer_sizes=(256, 96),
                epochs=35,
                batch_size=4096,
                class_weight="balanced",
                device="auto",
                random_state=random_state,
            )
        return models

    models = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "SVM Tuned": SVC(C=6.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state),
        "SVM Balanced": SVC(
            C=6.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Logistic Regression Tuned": LogisticRegression(C=2.0, max_iter=1000, random_state=random_state),
        "Logistic Regression Balanced": LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(200, 80),
            max_iter=700,
            early_stopping=True,
            random_state=random_state,
        ),
    }
    if _HAS_TORCH:
        models["Torch MLP"] = TorchMLPClassifier(
            hidden_layer_sizes=(256, 96),
            epochs=35,
            batch_size=4096,
            class_weight=None,
            device="auto",
            random_state=random_state,
        )
        models["Torch MLP Balanced"] = TorchMLPClassifier(
            hidden_layer_sizes=(256, 96),
            epochs=35,
            batch_size=4096,
            class_weight="balanced",
            device="auto",
            random_state=random_state,
        )
    return models


def _select_top_k_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    method: str = "mutual_info",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select top-k features using train-only statistics and transform train/test."""
    n_features = int(X_train.shape[1])
    if k <= 0 or k >= n_features:
        return X_train, X_test, np.arange(n_features)

    method_normalized = method.strip().lower()
    if method_normalized == "f_classif":
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method_normalized == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError("feature selection method must be 'mutual_info' or 'f_classif'")

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    selected_idx = selector.get_support(indices=True)
    return X_train_sel, X_test_sel, selected_idx


def _resolve_feature_pipeline_config(
    feature_selection_k: int | None,
    feature_selection_method: str,
    feature_pipeline_config: dict[str, Any] | None,
    default_scaler: str,
) -> dict[str, Any]:
    """Build effective feature-pipeline configuration with backward compatibility."""
    config = dict(feature_pipeline_config or {})

    if "scaler" not in config:
        config["scaler"] = default_scaler
    if feature_selection_k is not None and "selection_k" not in config:
        config["selection_k"] = int(feature_selection_k)
    if "selection_method" not in config:
        config["selection_method"] = feature_selection_method

    return config


def _select_uncorrelated_feature_indices(
    X_train: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Greedy correlation pruning on train-only data."""
    n_features = X_train.shape[1]
    if n_features <= 1:
        return np.arange(n_features)

    corr = np.corrcoef(X_train, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)

    keep = np.ones(n_features, dtype=bool)
    for i in range(n_features):
        if not keep[i]:
            continue
        to_drop = np.where(corr[i, i + 1 :] > threshold)[0] + (i + 1)
        keep[to_drop] = False

    return np.where(keep)[0]


def _apply_feature_pipeline_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    pipeline_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply scaling, filtering, selection and optional PCA per fold."""
    selected_idx = np.arange(X_train.shape[1])

    scaler_name = str(pipeline_config.get("scaler", "robust")).strip().lower()
    if scaler_name == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    X_train_t = scaler.fit_transform(X_train)
    X_test_t = scaler.transform(X_test)

    variance_threshold = pipeline_config.get("variance_threshold")
    if variance_threshold is not None:
        vt = VarianceThreshold(threshold=float(variance_threshold))
        X_train_t = vt.fit_transform(X_train_t)
        X_test_t = vt.transform(X_test_t)
        selected_idx = selected_idx[vt.get_support(indices=True)]

    corr_threshold = pipeline_config.get("correlation_threshold")
    if corr_threshold is not None:
        corr_threshold_float = float(corr_threshold)
        if 0.0 < corr_threshold_float < 1.0:
            keep_idx = _select_uncorrelated_feature_indices(X_train_t, corr_threshold_float)
            X_train_t = X_train_t[:, keep_idx]
            X_test_t = X_test_t[:, keep_idx]
            selected_idx = selected_idx[keep_idx]

    selection_k = pipeline_config.get("selection_k")
    if selection_k is not None:
        X_train_t, X_test_t, local_idx = _select_top_k_features(
            X_train=X_train_t,
            y_train=y_train,
            X_test=X_test_t,
            k=int(selection_k),
            method=str(pipeline_config.get("selection_method", "f_classif")),
        )
        selected_idx = selected_idx[local_idx]

    pca_variance = pipeline_config.get("pca_variance")
    pca_components = pipeline_config.get("pca_components")
    if pca_variance is not None or pca_components is not None:
        n_components = pca_components if pca_components is not None else pca_variance
        pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
        X_train_t = pca.fit_transform(X_train_t)
        X_test_t = pca.transform(X_test_t)

    return X_train_t, X_test_t, selected_idx


def _to_numpy(values: Any) -> np.ndarray:
    """Convert outputs from sklearn/cuML/cudf/cupy to a numpy array."""
    if isinstance(values, np.ndarray):
        return values

    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy())

    if hasattr(values, "get"):
        return np.asarray(values.get())

    return np.asarray(values)


def _fresh_estimator(model: Any) -> Any:
    """Create a fresh estimator instance for each fold/model run."""
    try:
        return clone(model)
    except Exception:
        if hasattr(model, "get_params"):
            return model.__class__(**model.get_params())
        return model.__class__()


def run_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    models: dict[str, Any] | None = None,
    show_progress: bool = True,
    backend: str = "auto",
    feature_selection_k: int | None = None,
    feature_selection_method: str = "mutual_info",
    feature_pipeline_config: dict[str, Any] | None = None,
    include_lstm: bool = False,
    lstm_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, list[Any]]]:
    """Run Leave-One-Subject-Out cross-validation across all provided models.

    Returns a nested dictionary per model with keys:
    - y_true: true labels across all folds
    - y_pred: predicted labels across all folds
    - y_prob: list of per-fold probability arrays (when available)
    - subject: subject id for each predicted sample

    If include_lstm=True, LSTM training is also run and merged into the same
    returned dictionary under the "LSTM" key.
    """
    resolved_backend = resolve_backend(backend)
    pipeline_config = _resolve_feature_pipeline_config(
        feature_selection_k=feature_selection_k,
        feature_selection_method=feature_selection_method,
        feature_pipeline_config=feature_pipeline_config,
        default_scaler="robust",
    )

    if models is None:
        models = get_default_models(backend=resolved_backend)

    subjects = sorted(np.unique(subject).tolist())
    results: dict[str, dict[str, list[Any]]] = {
        name: {
            "y_true": [],
            "y_pred": [],
            "y_prob": [],
            "subject": [],
            "selected_features": [],
        }
        for name in models
    }

    outer_iter = subjects
    if show_progress:
        outer_iter = tqdm(subjects, desc="LOSO folds", unit="subject")

    for test_subj in outer_iter:
        train_idx = subject != test_subj
        test_idx = subject == test_subj

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s, X_test_s, selected_idx = _apply_feature_pipeline_fold(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            pipeline_config=pipeline_config,
        )

        model_items = list(models.items())
        inner_iter = model_items
        if show_progress:
            inner_iter = tqdm(
                model_items,
                desc=f"Subject {test_subj}",
                leave=False,
                unit="model",
            )

        for name, model in inner_iter:
            clf = _fresh_estimator(model)
            clf.fit(X_train_s, y_train)
            y_pred = _to_numpy(clf.predict(X_test_s)).ravel()
            y_prob = _to_numpy(clf.predict_proba(X_test_s)) if hasattr(clf, "predict_proba") else None

            results[name]["y_true"].extend(y_test.tolist())
            results[name]["y_pred"].extend(y_pred.tolist())
            results[name]["subject"].extend([int(test_subj)] * len(y_test))
            results[name]["selected_features"].append(selected_idx.tolist())

            if y_prob is not None:
                results[name]["y_prob"].append(y_prob)

    if include_lstm:
        effective_lstm_config = dict(lstm_config or {})
        lstm_results = run_loso_lstm(
            X=X,
            y=y,
            subject=subject,
            feature_selection_k=feature_selection_k,
            feature_selection_method=feature_selection_method,
            feature_pipeline_config=pipeline_config,
            show_progress=show_progress,
            **effective_lstm_config,
        )
        if len(lstm_results.get("LSTM", {}).get("y_true", [])) > 0:
            results.update(lstm_results)

    return results


def summarize_selected_features(
    results: dict[str, dict[str, list[Any]]],
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Summarize how often each feature was selected across LOSO folds."""
    selections: list[list[int]] = []
    for model_result in results.values():
        if isinstance(model_result, dict) and model_result.get("selected_features"):
            selections = model_result["selected_features"]
            break

    if not selections:
        return pd.DataFrame(
            columns=["feature_index", "feature_name", "fold_count", "selection_rate"]
        )

    n_features = len(feature_names)
    fold_hits = np.zeros(n_features, dtype=int)
    for fold_selection in selections:
        fold_hits[np.asarray(fold_selection, dtype=int)] += 1

    selected_idx = np.where(fold_hits > 0)[0]
    if selected_idx.size == 0:
        return pd.DataFrame(
            columns=["feature_index", "feature_name", "fold_count", "selection_rate"]
        )

    n_folds = max(len(selections), 1)
    summary = pd.DataFrame(
        {
            "feature_index": selected_idx,
            "feature_name": [feature_names[i] for i in selected_idx],
            "fold_count": fold_hits[selected_idx],
            "selection_rate": fold_hits[selected_idx] / n_folds,
        }
    ).sort_values(["fold_count", "feature_index"], ascending=[False, True])

    return summary.head(top_n).reset_index(drop=True)


def build_subject_sequences(
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    sequence_length: int = 8,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-subject sliding-window sequences for sequence models."""
    if sequence_length <= 1:
        raise ValueError("sequence_length must be > 1")
    if stride <= 0:
        raise ValueError("stride must be >= 1")

    X_seq: list[np.ndarray] = []
    y_seq: list[int] = []
    s_seq: list[int] = []

    for sid in sorted(np.unique(subject).tolist()):
        idx = np.where(subject == sid)[0]
        X_sid = X[idx]
        y_sid = y[idx]

        if len(X_sid) < sequence_length:
            continue

        for start in range(0, len(X_sid) - sequence_length + 1, stride):
            end = start + sequence_length
            X_seq.append(X_sid[start:end])
            y_seq.append(int(y_sid[end - 1]))
            s_seq.append(int(sid))

    if not X_seq:
        return (
            np.empty((0, sequence_length, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.int64),
        np.asarray(s_seq, dtype=np.int64),
    )


def _resolve_torch_device(device: str = "auto") -> str:
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is unavailable. Install torch to run LSTM models. "
            f"Import error: {_TORCH_IMPORT_ERROR}"
        )

    device_normalized = device.strip().lower()
    if device_normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'")

    if device_normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_normalized == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_normalized


if _HAS_TORCH:
    class _LSTMSequenceClassifier(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_classes: int,
            dropout: float,
        ) -> None:
            super().__init__()
            recurrent_dropout = float(dropout) if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            last_hidden = out[:, -1, :]
            last_hidden = self.dropout(last_hidden)
            return self.classifier(last_hidden)


if _HAS_TORCH:
    class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
        """Sklearn-compatible tabular MLP backed by PyTorch/CUDA."""

        def __init__(
            self,
            hidden_layer_sizes: tuple[int, ...] = (256, 96),
            dropout: float = 0.2,
            epochs: int = 40,
            batch_size: int = 4096,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            class_weight: str | None = None,
            device: str = "auto",
            random_state: int = 42,
            verbose: bool = False,
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
            self.verbose = verbose

        def _build_model(self, input_size: int, n_classes: int):
            layers: list[Any] = []
            prev_size = input_size
            for hidden_size in self.hidden_layer_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(self.dropout)))
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, n_classes))
            return nn.Sequential(*layers)

        def fit(self, X: np.ndarray, y: np.ndarray):
            resolved_device = _resolve_torch_device(self.device)
            torch.manual_seed(int(self.random_state))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.random_state))

            X_np = np.asarray(X, dtype=np.float32)
            y_np = np.asarray(y)
            self.classes_ = np.asarray(sorted(np.unique(y_np).tolist()))
            label_to_idx = {label: idx for idx, label in enumerate(self.classes_.tolist())}
            y_idx = np.asarray([label_to_idx[label] for label in y_np], dtype=np.int64)

            dataset = TensorDataset(
                torch.tensor(X_np, dtype=torch.float32),
                torch.tensor(y_idx, dtype=torch.long),
            )
            generator = torch.Generator()
            generator.manual_seed(int(self.random_state))
            loader = DataLoader(
                dataset,
                batch_size=int(self.batch_size),
                shuffle=True,
                generator=generator,
                pin_memory=(resolved_device == "cuda"),
            )

            self.model_ = self._build_model(X_np.shape[1], len(self.classes_)).to(resolved_device)

            weight_tensor = None
            if self.class_weight == "balanced":
                counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(np.float32)
                weights = counts.sum() / (len(self.classes_) * np.maximum(counts, 1.0))
                weight_tensor = torch.tensor(weights, dtype=torch.float32).to(resolved_device)

            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            optimizer = torch.optim.AdamW(
                self.model_.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )

            self.model_.train()
            for epoch in range(int(self.epochs)):
                running_loss = 0.0
                for xb, yb in loader:
                    xb = xb.to(resolved_device, non_blocking=True)
                    yb = yb.to(resolved_device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model_(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.detach().cpu())
                if self.verbose:
                    print(f"TorchMLP epoch {epoch + 1}/{self.epochs}: loss={running_loss / max(len(loader), 1):.4f}")

            self.device_ = resolved_device
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            X_np = np.asarray(X, dtype=np.float32)
            self.model_.eval()
            probs_out: list[np.ndarray] = []
            loader = DataLoader(
                TensorDataset(torch.tensor(X_np, dtype=torch.float32)),
                batch_size=int(self.batch_size),
                shuffle=False,
                pin_memory=(self.device_ == "cuda"),
            )
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(self.device_, non_blocking=True)
                    logits = self.model_(xb)
                    probs_out.append(torch.softmax(logits, dim=1).cpu().numpy())
            return np.vstack(probs_out)

        def predict(self, X: np.ndarray) -> np.ndarray:
            probs = self.predict_proba(X)
            return self.classes_[np.argmax(probs, axis=1)]


def run_loso_lstm(
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    sequence_length: int = 8,
    stride: int = 1,
    hidden_size: int = 96,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    feature_selection_k: int | None = None,
    feature_selection_method: str = "mutual_info",
    feature_pipeline_config: dict[str, Any] | None = None,
    show_progress: bool = True,
    device: str = "auto",
    random_state: int = 42,
) -> dict[str, dict[str, list[Any]]]:
    """Run LOSO with an optional LSTM sequence classifier."""
    resolved_device = _resolve_torch_device(device)

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    class_labels = sorted(np.unique(y).tolist())
    label_to_idx = {int(label): idx for idx, label in enumerate(class_labels)}
    idx_to_label = np.asarray(class_labels)

    pipeline_config = _resolve_feature_pipeline_config(
        feature_selection_k=feature_selection_k,
        feature_selection_method=feature_selection_method,
        feature_pipeline_config=feature_pipeline_config,
        default_scaler="standard",
    )

    subjects = sorted(np.unique(subject).tolist())
    results = {
        "LSTM": {
            "y_true": [],
            "y_pred": [],
            "y_prob": [],
            "subject": [],
            "selected_features": [],
            "classes": class_labels,
        }
    }

    outer_iter = subjects
    if show_progress:
        outer_iter = tqdm(subjects, desc="LOSO folds (LSTM)", unit="subject")

    for test_subj in outer_iter:
        train_idx = subject != test_subj
        test_idx = subject == test_subj

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train_raw = y[train_idx]
        y_test_raw = y[test_idx]
        subject_train = subject[train_idx]
        subject_test = subject[test_idx]

        X_train_s, X_test_s, selected_idx = _apply_feature_pipeline_fold(
            X_train=X_train_raw,
            y_train=y_train_raw,
            X_test=X_test_raw,
            pipeline_config=pipeline_config,
        )

        X_train_seq, y_train_seq, _ = build_subject_sequences(
            X=X_train_s,
            y=y_train_raw,
            subject=subject_train,
            sequence_length=sequence_length,
            stride=stride,
        )
        X_test_seq, y_test_seq, _ = build_subject_sequences(
            X=X_test_s,
            y=y_test_raw,
            subject=subject_test,
            sequence_length=sequence_length,
            stride=stride,
        )

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            continue

        y_train_idx = np.asarray([label_to_idx[int(v)] for v in y_train_seq], dtype=np.int64)
        y_test_idx = np.asarray([label_to_idx[int(v)] for v in y_test_seq], dtype=np.int64)

        train_dataset = TensorDataset(
            torch.tensor(X_train_seq, dtype=torch.float32),
            torch.tensor(y_train_idx, dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = _LSTMSequenceClassifier(
            input_size=X_train_seq.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=len(class_labels),
            dropout=dropout,
        ).to(resolved_device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(resolved_device)
                yb = yb.to(resolved_device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            xb_test = torch.tensor(X_test_seq, dtype=torch.float32).to(resolved_device)
            logits = model(xb_test)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idx = probs.argmax(axis=1)

        y_pred_labels = idx_to_label[pred_idx]
        y_true_labels = idx_to_label[y_test_idx]

        results["LSTM"]["y_true"].extend(y_true_labels.tolist())
        results["LSTM"]["y_pred"].extend(y_pred_labels.tolist())
        results["LSTM"]["y_prob"].append(probs)
        results["LSTM"]["subject"].extend([int(test_subj)] * len(y_true_labels))
        results["LSTM"]["selected_features"].append(selected_idx.tolist())

    return results


def evaluate_feature_configs_loso(
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    probe_model: Any,
    configs: list[dict[str, Any]],
    backend: str = "auto",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Evaluate feature-pipeline configs with one probe model under LOSO."""
    rows: list[dict[str, Any]] = []

    config_iter = configs
    if show_progress:
        config_iter = tqdm(configs, desc="Feature config search", unit="config")

    for idx, config in enumerate(config_iter):
        config_name = str(config.get("name", f"config_{idx + 1}"))
        config_payload = {k: v for k, v in config.items() if k != "name"}

        probe_results = run_loso_cv(
            X=X,
            y=y,
            subject=subject,
            models={"Probe": probe_model},
            show_progress=False,
            backend=backend,
            feature_pipeline_config=config_payload,
            include_lstm=False,
        )

        metrics = compute_metrics_table(probe_results)
        if metrics.empty:
            continue

        metric_row = metrics.iloc[0]
        selected_sizes = [
            len(sel)
            for sel in probe_results["Probe"].get("selected_features", [])
            if isinstance(sel, list)
        ]
        mean_selected = float(np.mean(selected_sizes)) if selected_sizes else np.nan

        rows.append(
            {
                "config_name": config_name,
                "Accuracy": float(metric_row["Accuracy"]),
                "Precision": float(metric_row["Precision"]),
                "Recall": float(metric_row["Recall"]),
                "F1-Score": float(metric_row["F1-Score"]),
                "mean_selected_features": mean_selected,
                "config": config_payload,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "config_name",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "mean_selected_features",
                "config",
            ]
        )

    return pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)


def compute_metrics_table(
    results: dict[str, dict[str, list[Any]]],
    sort_by: str = "Macro F1",
) -> pd.DataFrame:
    """Compute robust model metrics and sort by the chosen primary metric.

    Weighted metrics are kept for backward compatibility, but Macro F1 and
    balanced accuracy are the safer headline metrics for imbalanced WESAD
    classes.
    """
    records: list[dict[str, Any]] = []

    for name, result in results.items():
        y_true = np.asarray(result["y_true"])
        y_pred = np.asarray(result["y_pred"])
        if len(y_true) == 0:
            continue

        weighted_precision = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        weighted_recall = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        records.append(
            {
                "Model": name,
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "Macro Precision": float(
                    precision_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "Macro Recall": float(
                    recall_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "Weighted Precision": weighted_precision,
                "Weighted Recall": weighted_recall,
                "Weighted F1": weighted_f1,
                # Backward-compatible aliases used by existing notebooks/scripts.
                "Precision": weighted_precision,
                "Recall": weighted_recall,
                "F1-Score": weighted_f1,
            }
        )

    metrics = pd.DataFrame(records)
    if metrics.empty:
        return metrics

    if sort_by in metrics.columns:
        metrics = metrics.sort_values(sort_by, ascending=False)
    return metrics.reset_index(drop=True)


def save_metrics_json(metrics_df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist the metrics table to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = metrics_df.to_dict(orient="records")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_loso_results(
    results: dict[str, dict[str, list[Any]]],
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
) -> None:
    """Persist LOSO results and feature data for later analysis (e.g. explainability).

    Stores y_true, y_pred, y_prob, subject, feature_names etc. so that
    05-explainability.ipynb can load them without re-running training.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())
    y_true_list: list[list[int]] = []
    y_pred_list: list[list[int]] = []
    subject_list: list[list[int]] = []
    y_prob_list: list[list[float]] = []

    for name in model_names:
        r = results[name]
        y_true_list.append(r["y_true"])
        y_pred_list.append(r["y_pred"])
        subject_list.append(r["subject"])
        y_prob_list.append(np.vstack(r["y_prob"]).tolist() if r.get("y_prob") else [])

    np.savez_compressed(
        path,
        model_names=np.array(model_names, dtype=object),
        y_true=np.array(y_true_list, dtype=object),
        y_pred=np.array(y_pred_list, dtype=object),
        subject=np.array(subject_list, dtype=object),
        y_prob=np.array(y_prob_list, dtype=object),
        X=X,
        y=y,
        subject_global=subject,
        feature_names=np.array(feature_names, dtype=object),
    )
    print(f"Saved LOSO results to: {path}.npz")


def load_loso_results(
    output_path: str | Path,
) -> dict[str, Any]:
    """Reconstruct results dict from .npz saved by save_loso_results."""
    path = Path(str(output_path).removesuffix(".npz"))
    data = np.load(f"{path}.npz", allow_pickle=True)

    model_names: list[str] = data["model_names"].tolist()
    results: dict[str, dict[str, list[Any]]] = {}
    for i, name in enumerate(model_names):
        y_prob_arr: list[np.ndarray] = []
        raw = data["y_prob"][i]
        if isinstance(raw, np.ndarray) and raw.ndim > 0 and len(raw) > 0:
            y_prob_arr = [np.array(p) for p in raw] if raw.dtype == object else [raw]
        results[name] = {
            "y_true": [int(v) for v in data["y_true"][i]],
            "y_pred": [int(v) for v in data["y_pred"][i]],
            "subject": [int(v) for v in data["subject"][i]],
            "y_prob": y_prob_arr,
        }

    return {
        "results": results,
        "X": data["X"],
        "y": data["y"],
        "subject": data["subject_global"],
        "feature_names": data["feature_names"].tolist(),
    }
