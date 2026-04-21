"""Utilities for loading WESAD subject data from the local raw dataset."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "01_raw" / "WESAD"
SUBJECT_PATTERN = re.compile(r"^S\d+$")


@dataclass(slots=True)
class SubjectData:
    """Normalized wrist-only representation of one WESAD subject."""

    subject_id: str
    subject_info: Any
    labels: np.ndarray
    wrist_signals: dict[str, np.ndarray]
    source_path: Path


def normalize_subject_id(subject: int | str) -> str:
    """Convert an integer or string into the canonical WESAD subject id."""
    if isinstance(subject, int):
        return f"S{subject}"

    cleaned = str(subject).strip().upper()
    if cleaned.isdigit():
        return f"S{cleaned}"
    if SUBJECT_PATTERN.fullmatch(cleaned):
        return cleaned

    raise ValueError(f"Invalid subject identifier: {subject!r}")


def list_subjects(raw_dir: str | Path = DEFAULT_RAW_DIR) -> list[str]:
    """Return sorted subject ids available under the raw WESAD directory."""
    raw_path = Path(raw_dir)
    subjects = [
        item.name
        for item in raw_path.iterdir()
        if item.is_dir() and SUBJECT_PATTERN.fullmatch(item.name)
    ]
    return sorted(subjects, key=lambda value: int(value[1:]))


def get_subject_path(subject: int | str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    """Return the expected pickle path for a subject."""
    subject_id = normalize_subject_id(subject)
    subject_path = Path(raw_dir) / subject_id / f"{subject_id}.pkl"
    if not subject_path.exists():
        raise FileNotFoundError(f"Subject file not found: {subject_path}")
    return subject_path


def load_subject_pickle(subject: int | str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> dict[str, Any]:
    """Load the original WESAD pickle for a subject."""
    subject_path = get_subject_path(subject, raw_dir=raw_dir)
    with subject_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_subject(subject: int | str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> SubjectData:
    """Load one subject and keep only the wrist signals used by this project."""
    payload = load_subject_pickle(subject, raw_dir=raw_dir)
    subject_id = normalize_subject_id(subject)

    wrist_signals = {
        signal_name: np.asarray(values)
        for signal_name, values in payload["signal"]["wrist"].items()
    }

    return SubjectData(
        subject_id=subject_id,
        subject_info=payload.get("subject"),
        labels=np.asarray(payload["label"]),
        wrist_signals=wrist_signals,
        source_path=get_subject_path(subject_id, raw_dir=raw_dir),
    )


def describe_subject(subject: int | str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> dict[str, Any]:
    """Return a compact summary useful for EDA and sanity checks."""
    subject_data = load_subject(subject, raw_dir=raw_dir)
    return {
        "subject_id": subject_data.subject_id,
        "source_path": str(subject_data.source_path),
        "labels_shape": list(subject_data.labels.shape),
        "wrist_signals": {
            name: {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
            for name, values in subject_data.wrist_signals.items()
        },
    }


def load_all_subjects(
    subjects: list[int | str] | None = None,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> list[SubjectData]:
    """Load multiple subjects using the same normalized structure."""
    subject_ids = subjects or list_subjects(raw_dir=raw_dir)
    return [load_subject(subject, raw_dir=raw_dir) for subject in subject_ids]

