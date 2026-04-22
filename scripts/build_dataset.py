"""Build processed dataset from raw WESAD wrist data.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --config config.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import process_all_subjects
from src.features import extract_features_from_windows
from src.loader import list_subjects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the processed feature dataset from raw WESAD wrist data."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "01_raw" / "WESAD",
        help="Path to the raw WESAD directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "03_processed",
        help="Directory to write the output files.",
    )
    args = parser.parse_args()

    subjects = sorted(list_subjects(raw_dir=args.raw_dir), key=lambda s: int(s[1:]))
    target_labels = {1, 2, 3}

    print(f"Raw directory: {args.raw_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Subjects: {subjects}")
    print(f"Target labels: {target_labels}")
    print("-" * 40)

    start = time.time()
    subject_windows = process_all_subjects(
        subjects=subjects,
        raw_dir=args.raw_dir,
        target_labels=target_labels,
    )
    print(f"Windowed {len(subject_windows)} subjects in {time.time() - start:.1f}s")

    start = time.time()
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    subject_indices: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for sw in subject_windows:
        X, y, names = extract_features_from_windows(sw)
        if feature_names is None:
            feature_names = names
        all_X.append(X)
        all_y.append(y)
        subject_indices.append(np.full(len(y), int(sw.subject_id[1:])))
        print(f"  {sw.subject_id}: {len(y)} windows, features {X.shape[1]}")

    X_all = np.vstack(all_X)
    y_all = np.hstack(all_y)
    subj_all = np.hstack(subject_indices)

    elapsed = time.time() - start
    print(f"Extracted features in {elapsed:.1f}s")
    print(f"Total samples: {X_all.shape[0]}")
    print(f"Feature dimension: {X_all.shape[1]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "features.npz",
        X=X_all,
        y=y_all,
        subject=subj_all,
        feature_names=np.array(feature_names, dtype=object),
    )
    print(f"Saved to: {args.output_dir / 'features.npz'}")


if __name__ == "__main__":
    main()
