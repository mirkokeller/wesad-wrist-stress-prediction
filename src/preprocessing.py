"""Preprocessing for WESAD wrist signals: filtering, windowing per signal type.

Follows the paper's approach from Section 4.1:
- Window shift: 0.25 s for all signals
- ACC window: 5 s
- BVP / EDA / TEMP window: 60 s
- Signals are kept at native sampling rates for downstream feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from src.loader import SubjectData, list_subjects, load_subject, DEFAULT_RAW_DIR

# ── Native sampling rates ------------------------------------------------------
SIGNAL_HZ = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
}
LABEL_HZ = 700
SHIFT_S = 0.25


# ── Basic filters --------------------------------------------------------------

def lowpass_filter(
    values: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = scipy_signal.butter(order, normal_cutoff, btype="low")
    return scipy_signal.filtfilt(b, a, values, axis=0)


def bandpass_filter(
    values: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, values, axis=0)


# ── Preprocessing per signal ---------------------------------------------------

def preprocess_acc(acc: np.ndarray, fs: float = 32.0) -> np.ndarray:
    """ACC receives no explicit filter in the paper (raw ACC features used)."""
    return acc


def preprocess_bvp(bvp: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """BVP is kept at 64 Hz for peak detection."""
    return bvp


def preprocess_eda(eda: np.ndarray, fs: float = 4.0) -> np.ndarray:
    """EDA preprocessing: keep raw signal.

    The paper mentions 5 Hz lowpass, but EDA is sampled at 4 Hz where the
    Nyquist frequency is 2 Hz, making a 5 Hz filter impossible.
    We use raw EDA as-in and compute features directly, which is valid
    since statistical features (mean, std, range) don't require filtering.
    """
    return eda


def preprocess_temp(temp: np.ndarray, fs: float = 4.0) -> np.ndarray:
    """TEMP receives no explicit filter."""
    return temp


def preprocess_subject(subject: SubjectData) -> dict[str, np.ndarray]:
    """Apply signal-specific filtering to all wrist signals of a subject."""
    return {
        "ACC": preprocess_acc(subject.wrist_signals["ACC"], fs=SIGNAL_HZ["ACC"]),
        "BVP": preprocess_bvp(subject.wrist_signals["BVP"], fs=SIGNAL_HZ["BVP"]),
        "EDA": preprocess_eda(subject.wrist_signals["EDA"], fs=SIGNAL_HZ["EDA"]),
        "TEMP": preprocess_temp(subject.wrist_signals["TEMP"], fs=SIGNAL_HZ["TEMP"]),
        "labels": subject.labels.astype(int).ravel(),
    }


# ── Windowing ------------------------------------------------------------------

def _label_mode_per_block(labels: np.ndarray, block_size: int) -> np.ndarray:
    """Downsample integer labels by block-wise mode.

    Parameters
    ----------
    labels : np.ndarray shape (N,)
        Original 700 Hz labels.
    block_size : int
        Samples per block (e.g. 175 for 0.25 s at 700 Hz).

    Returns
    -------
    np.ndarray shape (M,)
        Mode label of each block.
    """
    n_blocks = labels.shape[0] // block_size
    truncated = labels[: n_blocks * block_size]
    blocks = truncated.reshape(n_blocks, block_size)
    # bincount-based mode (labels are small non-negative ints)
    # Tie-break with the last value in the block to better preserve
    # immediate temporal continuity at transitions.
    modes = np.empty(n_blocks, dtype=int)
    for i in range(n_blocks):
        block = blocks[i]
        counts = np.bincount(block)
        max_count = int(np.max(counts))
        winners = np.where(counts == max_count)[0]
        if winners.size == 1:
            modes[i] = int(winners[0])
        else:
            modes[i] = int(block[-1])
    return modes


def label_grid_4hz(labels: np.ndarray, label_hz: float = 700.0) -> np.ndarray:
    """Convert 700 Hz labels to a 4 Hz (0.25 s) grid via block-wise mode."""
    block_size = int(round(label_hz / 4.0))  # 175
    if block_size == 1:
        return labels
    return _label_mode_per_block(labels, block_size)


@dataclass(slots=True)
class SubjectWindows:
    """Windows extracted from one subject, ready for feature extraction."""

    subject_id: str
    acc_windows: dict[str, np.ndarray]  # shape (N, 5*32, 3) = (N, 160, 3)
    bvp_windows: np.ndarray             # shape (N, 60*64, 1) = (N, 3840, 1)
    eda_windows: np.ndarray             # shape (N, 60*4, 1)  = (N, 240, 1)
    temp_windows: np.ndarray            # shape (N, 60*4, 1)  = (N, 240, 1)
    labels: np.ndarray                  # shape (N,)


def create_windows_for_subject(
    subject: SubjectData,
    shift_s: float = SHIFT_S,
    acc_win_s: float = 5.0,
    physio_win_s: float = 60.0,
    target_labels: set[int] | None = None,
    require_pure_physio_window: bool = True,
    transition_buffer_s: float = 0.0,
) -> SubjectWindows | None:
    """Build sliding windows for one subject following the paper's segmentation.

    - All signals share the same 0.25 s time grid.
    - Windows are causal: [t - win_size, t].
    - The first usable window ends at ``physio_win_s``.
    - By default, the whole physiological window must belong to the same
      protocol label. This avoids training on a 60 s feature window whose
      label only describes the final instant after a condition transition.
    """
    target_labels = target_labels or {1, 2, 3}
    ps = preprocess_subject(subject)

    # Determine unified recording length in seconds from the minimum
    # available track to keep all modalities and labels aligned.
    rec_len_s = min(
        ps["labels"].shape[0] / LABEL_HZ,
        ps["ACC"].shape[0] / SIGNAL_HZ["ACC"],
        ps["BVP"].shape[0] / SIGNAL_HZ["BVP"],
        ps["EDA"].shape[0] / SIGNAL_HZ["EDA"],
        ps["TEMP"].shape[0] / SIGNAL_HZ["TEMP"],
    )

    # --- Build the 4 Hz time axis --------------------------------------------
    # Each step = 0.25 s. First usable physio window ends at t = 60 s.
    # Steps go from physio_win_s to rec_len_s with step = shift_s
    step_count = int(np.floor((rec_len_s - physio_win_s) / shift_s)) + 1
    if step_count <= 0:
        return None

    # Window sizes in samples
    acc_n = int(acc_win_s * SIGNAL_HZ["ACC"])   # 160
    bvp_n = int(physio_win_s * SIGNAL_HZ["BVP"])  # 3840
    eda_n = int(physio_win_s * SIGNAL_HZ["EDA"])  # 240
    temp_n = int(physio_win_s * SIGNAL_HZ["TEMP"])  # 240

    acc_w = []
    bvp_w = []
    eda_w = []
    temp_w = []
    label_w = []

    labels_4hz = label_grid_4hz(ps["labels"])

    for i in range(step_count):
        t_end = physio_win_s + i * shift_s
        # Label index on the 4 Hz grid at the window end.
        # Use the last completed 0.25 s block (causal), not the next block.
        label_idx = int(round(t_end / shift_s)) - 1
        if label_idx >= labels_4hz.shape[0]:
            break
        if label_idx < 0:
            continue
        lbl = labels_4hz[label_idx]
        if lbl not in target_labels:
            continue

        phys_label_start_idx = int(round((t_end - physio_win_s) / shift_s))
        phys_label_end_idx = label_idx + 1
        if phys_label_start_idx < 0:
            continue

        if require_pure_physio_window:
            label_slice = labels_4hz[phys_label_start_idx:phys_label_end_idx]
            if label_slice.size == 0 or not np.all(label_slice == lbl):
                continue

        if transition_buffer_s > 0:
            buffer_blocks = int(round(transition_buffer_s / shift_s))
            buffer_start_idx = max(0, label_idx - buffer_blocks + 1)
            recent_labels = labels_4hz[buffer_start_idx:phys_label_end_idx]
            if recent_labels.size == 0 or not np.all(recent_labels == lbl):
                continue

        # --- ACC window (5 s ending at t_end) --------------------------------
        acc_start_s = t_end - acc_win_s
        acc_start = int(round(acc_start_s * SIGNAL_HZ["ACC"]))
        acc_end = int(round(t_end * SIGNAL_HZ["ACC"]))
        if acc_end > ps["ACC"].shape[0]:
            break
        acc_w.append(ps["ACC"][acc_start:acc_end])

        # --- Physiological window (60 s ending at t_end) -----------------------
        phys_start_s = t_end - physio_win_s
        bvp_start = int(round(phys_start_s * SIGNAL_HZ["BVP"]))
        bvp_end = int(round(t_end * SIGNAL_HZ["BVP"]))
        if bvp_end > ps["BVP"].shape[0]:
            break
        bvp_w.append(ps["BVP"][bvp_start:bvp_end])

        eda_start = int(round(phys_start_s * SIGNAL_HZ["EDA"]))
        eda_end = int(round(t_end * SIGNAL_HZ["EDA"]))
        if eda_end > ps["EDA"].shape[0]:
            break
        eda_w.append(ps["EDA"][eda_start:eda_end])

        temp_start = int(round(phys_start_s * SIGNAL_HZ["TEMP"]))
        temp_end = int(round(t_end * SIGNAL_HZ["TEMP"]))
        if temp_end > ps["TEMP"].shape[0]:
            break
        temp_w.append(ps["TEMP"][temp_start:temp_end])

        label_w.append(lbl)

    if not label_w:
        return None

    return SubjectWindows(
        subject_id=subject.subject_id,
        acc_windows={"raw": np.stack(acc_w, axis=0)},
        bvp_windows=np.stack(bvp_w, axis=0),
        eda_windows=np.stack(eda_w, axis=0),
        temp_windows=np.stack(temp_w, axis=0),
        labels=np.array(label_w, dtype=int),
    )


def process_all_subjects(
    subjects: list[int | str] | None = None,
    raw_dir: str | Path | None = None,
    target_labels: set[int] | None = None,
    shift_s: float = SHIFT_S,
    acc_win_s: float = 5.0,
    physio_win_s: float = 60.0,
    require_pure_physio_window: bool = True,
    transition_buffer_s: float = 0.0,
) -> list[SubjectWindows]:
    """Run windowing on multiple subjects."""
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    if subjects is None:
        subjects = list_subjects(raw_dir=raw_dir)
    results = []
    for sid in subjects:
        raw = load_subject(sid, raw_dir=raw_dir)
        windows = create_windows_for_subject(
            raw,
            shift_s=shift_s,
            acc_win_s=acc_win_s,
            physio_win_s=physio_win_s,
            target_labels=target_labels,
            require_pure_physio_window=require_pure_physio_window,
            transition_buffer_s=transition_buffer_s,
        )
        if windows is not None:
            results.append(windows)
    return results
