"""Build processed dataset from raw WESAD wrist data.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --allow-mixed-label-windows
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
from src.loader import list_subjects, resolve_raw_dir


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
    parser.add_argument(
        "--acc-window-sec",
        type=float,
        default=5.0,
        help="ACC window length in seconds.",
    )
    parser.add_argument(
        "--physio-window-sec",
        type=float,
        default=60.0,
        help="BVP/EDA/TEMP window length in seconds.",
    )
    parser.add_argument(
        "--shift-sec",
        type=float,
        default=0.25,
        help="Sliding-window shift in seconds.",
    )
    parser.add_argument(
        "--transition-buffer-sec",
        type=float,
        default=0.0,
        help="Extra time after a label transition to discard, in seconds.",
    )
    parser.add_argument(
        "--allow-mixed-label-windows",
        action="store_true",
        help=(
            "Allow 60 s physiological windows that contain multiple protocol "
            "labels. By default they are discarded to reduce transition leakage."
        ),
    )
    args = parser.parse_args()

    resolved_raw_dir = resolve_raw_dir(args.raw_dir)

    subjects = sorted(list_subjects(raw_dir=resolved_raw_dir), key=lambda s: int(s[1:]))
    target_labels = {1, 2, 3}

    print(f"Raw directory: {args.raw_dir}")
    print(f"Resolved raw directory: {resolved_raw_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Subjects: {subjects}")
    print(f"Target labels: {target_labels}")
    print(f"Windowing: ACC={args.acc_window_sec}s, physio={args.physio_window_sec}s, shift={args.shift_sec}s")
    print(f"Require pure physio labels: {not args.allow_mixed_label_windows}")
    print(f"Transition buffer: {args.transition_buffer_sec}s")
    print("-" * 40)

    start = time.time()
    subject_windows = process_all_subjects(
        subjects=subjects,
        raw_dir=resolved_raw_dir,
        target_labels=target_labels,
        shift_s=args.shift_sec,
        acc_win_s=args.acc_window_sec,
        physio_win_s=args.physio_window_sec,
        require_pure_physio_window=not args.allow_mixed_label_windows,
        transition_buffer_s=args.transition_buffer_sec,
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
        window_policy=np.array(
            {
                "acc_window_sec": args.acc_window_sec,
                "physio_window_sec": args.physio_window_sec,
                "shift_sec": args.shift_sec,
                "require_pure_physio_window": not args.allow_mixed_label_windows,
                "transition_buffer_sec": args.transition_buffer_sec,
            },
            dtype=object,
        ),
    )
    print(f"Saved to: {args.output_dir / 'features.npz'}")


if __name__ == "__main__":
    main()
