"""Feature extraction for WESAD wrist signals per Section 4.1 of the WESAD paper.

Implements the main feature groups for wrist modalities: ACC, BVP, EDA, TEMP.
Excludes chest signals (ECG, EMG, RESP) as required.

For BVP, a manual peak-detection pipeline is used so that NeuroKit2 is not required.
For EDA, a manual smooth-baseline (tonic) / residual (phasic) approach is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats


# ── ACC features ----------------------------------------------------------------

def acc_features(acc_window: np.ndarray, fs: float = 32.0) -> dict[str, float]:
    """Extract statistical and spectral features from a 5-second ACC window.

    Parameters
    ----------
    acc_window : np.ndarray shape (samples, 3)
        One 5-second ACC window (x, y, z).
    fs : float, default 32.0
        Sampling rate of ACC.

    Returns
    -------
    dict[str, float]
        Feature dictionary.
    """
    features: dict[str, float] = {}
    axes = ["x", "y", "z"]

    for i, axis in enumerate(axes):
        a = acc_window[:, i]
        features[f"acc_{axis}_mean"] = float(np.mean(a))
        features[f"acc_{axis}_std"] = float(np.std(a, ddof=1))
        features[f"acc_{axis}_min"] = float(np.min(a))
        features[f"acc_{axis}_max"] = float(np.max(a))
        features[f"acc_{axis}_range"] = float(np.max(a) - np.min(a))

        # Peak frequency via FFT
        n = len(a)
        fft_vals = np.fft.rfft(a)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        magnitudes = np.abs(fft_vals)
        peak_idx = int(np.argmax(magnitudes))
        features[f"acc_{axis}_peak_freq"] = float(freqs[peak_idx])

    # Magnitude over 3 axes
    mag = np.sqrt(np.sum(acc_window ** 2, axis=1))
    features["acc_mag_mean"] = float(np.mean(mag))
    features["acc_mag_std"] = float(np.std(mag, ddof=1))
    features["acc_mag_min"] = float(np.min(mag))
    features["acc_mag_max"] = float(np.max(mag))
    features["acc_mag_range"] = float(np.max(mag) - np.min(mag))

    # Magnitude peak frequency
    n = len(mag)
    fft_vals = np.fft.rfft(mag)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitudes = np.abs(fft_vals)
    peak_idx = int(np.argmax(magnitudes))
    features["acc_mag_peak_freq"] = float(freqs[peak_idx])

    return features


# ── BVP features (manual peak detection) ----------------------------------------

def _find_peaks_simple(signal: np.ndarray, fs: float, min_distance_s: float = 0.3) -> np.ndarray:
    """Simple peak detector for PPG: find positive zero-crossings of first derivative.

    Parameters
    ----------
    signal : np.ndarray
        PPG signal segment.
    fs : float
        Sampling rate.
    min_distance_s : float, default 0.3
        Minimum distance between peaks in seconds.

    Returns
    -------
    np.ndarray
        Peak sample indices.
    """
    # Center the signal
    sig = signal.ravel() - np.mean(signal)

    # First derivative
    dx = np.diff(sig)
    # Zero-crossings: where derivative goes positive (minima -> maxima)
    # Actually we want upward crossing: dx[:-1] < 0 and dx[1:] > 0
    # and the original signal is positive around that region
    zero_cross = np.where((dx[:-1] < 0) & (dx[1:] >= 0))[0] + 1

    # Refine: find local maxima around each zero-crossing
    peaks = []
    search_radius = int(0.15 * fs)  # 150 ms search window
    for zc in zero_cross:
        start = max(0, zc - search_radius)
        end = min(len(sig), zc + search_radius)
        local_max = start + int(np.argmax(sig[start:end]))
        peaks.append(local_max)

    if not peaks:
        return np.array([], dtype=int)

    peaks = np.array(peaks, dtype=int)

    # Minimum distance filter
    min_samples = int(min_distance_s * fs)
    if min_samples <= 1:
        return peaks

    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_samples:
            filtered.append(p)
    return np.array(filtered, dtype=int)


def bvp_features(bvp_window: np.ndarray, fs: float = 64.0) -> dict[str, float]:
    """Extract HR and HRV features from a 60-second BVP window.

    HR: heart rate in bpm
    HRV: NN intervals + frequency-domain energy (ULF, LF, HF, UHF)
    """
    features: dict[str, float] = {}
    sig = bvp_window.ravel()

    if len(sig) == 0:
        return {k: 0.0 for k in _BVP_FEATURE_KEYS}

    # Find peaks
    peaks = _find_peaks_simple(sig, fs=fs)

    if len(peaks) < 2:
        return {k: 0.0 for k in _BVP_FEATURE_KEYS}

    # HR from inter-peak intervals (in seconds)
    peak_diffs = np.diff(peaks) / fs
    hr = 60.0 / peak_diffs
    features["bvp_hr_mean"] = float(np.mean(hr))
    features["bvp_hr_std"] = float(np.std(hr, ddof=1))

    # NN intervals: use peak intervals directly (in seconds)
    nn_intervals = peak_diffs
    features["bvp_hrv_mean_nn"] = float(np.mean(nn_intervals))
    features["bvp_hrv_std_nn"] = float(np.std(nn_intervals, ddof=1))
    features["bvp_hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(nn_intervals) ** 2)))

    # Frequency-domain HRV on NN intervals (resample to 4 Hz for spectral estimation)
    if len(nn_intervals) >= 4:
        # Cumulative sum of NN intervals -> time stamps
        time_stamps = np.cumsum(nn_intervals)
        total_time = time_stamps[-1]
        # Resample NN interval series to 4 Hz
        hrv_fs = 4.0
        num_hrv_samples = int(total_time * hrv_fs)
        if num_hrv_samples > 1:
            hrv_time_uniform = np.linspace(0, total_time, num_hrv_samples)
            hrv_signal = np.interp(hrv_time_uniform, time_stamps, nn_intervals)

            # Welch PSD with reduced computation
            nperseg = min(64, num_hrv_samples)  # Reduced from 128
            if nperseg < 16:  # Minimum segment size
                nperseg = min(16, num_hrv_samples)
            if nperseg > 0:
                f, psd = scipy_signal.welch(
                    hrv_signal - np.mean(hrv_signal),
                    fs=hrv_fs,
                    nperseg=nperseg,
                    noverlap=nperseg//2,  # 50% overlap
                    window="hann",
                )
                df = f[1] - f[0] if len(f) > 1 else 1.0

                # Power in bands
                def _band_power(low: float, high: float) -> float:
                    mask = (f >= low) & (f < high)
                    return float(np.sum(psd[mask]) * df) if len(psd) > 0 else 0.0
            else:
                # If we don't have sufficient samples, return zeros
                def _band_power(low: float, high: float) -> float:
                    return 0.0

            features["bvp_hrv_ulf"] = _band_power(0.01, 0.04)
            features["bvp_hrv_lf"] = _band_power(0.04, 0.15)
            features["bvp_hrv_hf"] = _band_power(0.15, 0.4)
            features["bvp_hrv_uhf"] = _band_power(0.4, 1.0)
        else:
            for band in ["ulf", "lf", "hf", "uhf"]:
                features[f"bvp_hrv_{band}"] = 0.0
    else:
        for band in ["ulf", "lf", "hf", "uhf"]:
            features[f"bvp_hrv_{band}"] = 0.0

    return features


_BVP_FEATURE_KEYS = [
    "bvp_hr_mean",
    "bvp_hr_std",
    "bvp_hrv_mean_nn",
    "bvp_hrv_std_nn",
    "bvp_hrv_rmssd",
    "bvp_hrv_ulf",
    "bvp_hrv_lf",
    "bvp_hrv_hf",
    "bvp_hrv_uhf",
]


# ── EDA features ----------------------------------------------------------------

def eda_features(eda_window: np.ndarray, fs: float = 4.0) -> dict[str, float]:
    """Extract statistical and SCL/SCR features from a 60-second EDA window.

    Follows the paper approach: separate tonic (SCL) and phasic (SCR).
    Tonic = moving median (slow baseline).
    Phasic = EDA - tonic.
    #SCR = number of peaks in the phasic component.
    """
    features: dict[str, float] = {}
    sig = eda_window.ravel()

    if len(sig) == 0:
        return {k: 0.0 for k in _EDA_FEATURE_KEYS}

    # Basic stats
    features["eda_mean"] = float(np.mean(sig))
    features["eda_std"] = float(np.std(sig, ddof=1))
    features["eda_min"] = float(np.min(sig))
    features["eda_max"] = float(np.max(sig))
    features["eda_range"] = float(np.max(sig) - np.min(sig))

    # Tonic / SCL component via moving median (slow baseline, ~30s smoothing)
    kernel_size = min(int(30.0 * fs), len(sig))
    # median filter requires odd kernel size
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        kernel_size = 3
    scl = scipy_signal.medfilt(sig, kernel_size=kernel_size)
    scl_mean = float(np.mean(scl))
    scl_std = float(np.std(scl, ddof=1))
    features["eda_scl_mean"] = scl_mean
    features["eda_scl_std"] = scl_std

    # Phasic / SCR component
    scr = sig - scl
    scr_mean = float(np.mean(scr))
    scr_std = float(np.std(scr, ddof=1))
    features["eda_scr_mean"] = scr_mean
    features["eda_scr_std"] = scr_std

    # Count SCR peaks in phasic component
    # Simple threshold-based peak detection on the phasic component
    threshold = scr_mean + 0.5 * scr_std if scr_std > 0 else scr_mean
    scr_positive = scr > threshold
    # Peaks = transitions from not-peak to peak
    peaks = np.where((scr_positive[:-1] == False) & (scr_positive[1:] == True))[0]
    features["eda_num_scr_peaks"] = float(len(peaks))

    return features


_EDA_FEATURE_KEYS = [
    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",
    "eda_range",
    "eda_scl_mean",
    "eda_scl_std",
    "eda_scr_mean",
    "eda_scr_std",
    "eda_num_scr_peaks",
]


# ── TEMP features ---------------------------------------------------------------

def temp_features(temp_window: np.ndarray) -> dict[str, float]:
    """Extract statistical features from a 60-second TEMP window."""
    sig = temp_window.ravel()
    return {
        "temp_mean": float(np.mean(sig)),
        "temp_std": float(np.std(sig, ddof=1)),
        "temp_min": float(np.min(sig)),
        "temp_max": float(np.max(sig)),
        "temp_range": float(np.max(sig) - np.min(sig)),
    }


# ── Combined pipeline -----------------------------------------------------------

def extract_features_from_windows(windows) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature vectors and labels from a SubjectWindows object.

    Parameters
    ----------
    windows : SubjectWindows
        Output from preprocessing.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        Feature matrix shape (N, D), label array shape (N,), and feature names list.
    """
    rows: list[dict[str, float]] = []

    n = len(windows.labels)
    for i in range(n):
        acc = windows.acc_windows["raw"][i]
        bvp = windows.bvp_windows[i]
        eda = windows.eda_windows[i]
        temp = windows.temp_windows[i]

        feats: dict[str, float] = {}
        feats.update(acc_features(acc))
        feats.update(bvp_features(bvp))
        feats.update(eda_features(eda))
        feats.update(temp_features(temp))
        rows.append(feats)

    # Build dataframe-like matrix
    all_keys = sorted(rows[0].keys())
    X = np.array([[r[k] for k in all_keys] for r in rows], dtype=float)
    y = windows.labels
    return X, y, all_keys
