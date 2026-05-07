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


# ── Helper functions -----------------------------------------------------------

def _sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Compute Sample Entropy (SampEn) of a signal.

    Sample Entropy measures the regularity/complexity of a signal.
    Lower values = more regular/predictable (often associated with stress).
    Higher values = more complex/unpredictable (often associated with relaxation).

    Parameters
    ----------
    signal : np.ndarray
        Input signal (e.g., NN intervals).
    m : int, default 2
        Embedding dimension (template length).
    r : float, default 0.2
        Tolerance (fraction of std).

    Returns
    -------
    float
        Sample Entropy value.
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)

    if n < m + 1:
        return 0.0

    # Compute standard deviation for tolerance
    std = np.std(sig, ddof=0)
    if std == 0:
        return 0.0

    threshold = r * std

    # Build template vectors
    def _count_matches(template: np.ndarray, all_vectors: np.ndarray) -> int:
        """Count vectors within threshold of template."""
        diffs = np.abs(all_vectors - template)
        return int(np.sum(np.max(diffs, axis=1) < threshold))

    # Extract overlapping m-length vectors
    x_m = np.array([sig[i:i+m] for i in range(n - m)])
    x_m1 = np.array([sig[i:i+m+1] for i in range(n - m)])

    # Count matches
    match_m = sum(_count_matches(x_m[i], x_m[i+1:]) for i in range(len(x_m)))
    match_m1 = sum(_count_matches(x_m1[i], x_m1[i+1:]) for i in range(len(x_m1)))

    # Avoid division by zero
    total_pairs = len(x_m) * (len(x_m) - 1) / 2
    if total_pairs == 0 or match_m == 0:
        return 0.0

    sampen = -np.log(match_m1 / match_m) if match_m1 > 0 else 0.0
    return float(sampen)


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
        if len(magnitudes) > 0:
            magnitudes[0] = 0.0
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
    if len(magnitudes) > 0:
        magnitudes[0] = 0.0
    peak_idx = int(np.argmax(magnitudes))
    features["acc_mag_peak_freq"] = float(freqs[peak_idx])

    return features


# ── BVP features (manual peak detection) ----------------------------------------

def _find_peaks_improved(
    signal: np.ndarray,
    fs: float,
    min_distance_s: float = 0.3,
) -> tuple[np.ndarray, int]:
    """Improved peak detector for PPG using scipy's find_peaks with adaptive thresholds.

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
    sig = signal.ravel()

    if len(sig) < 2:
        return np.array([], dtype=int), 0

    # Preprocess: remove baseline drift using detrending
    from scipy import signal as scipy_signal
    detrended = scipy_signal.detrend(sig, type='linear')

    # Compute adaptive threshold based on signal statistics
    median_val = np.median(detrended)
    mad = np.median(np.abs(detrended - median_val)) * 1.4826  # Robust estimate of std

    # Set height threshold: peaks should be at least 1 MAD above median
    height = median_val + 0.5 * mad

    # Set prominence threshold: peaks should stand out from local noise
    # Prominence is the height of the peak above its surrounding baseline
    prominence = 0.3 * mad

    # Set width constraint based on expected heart rate range (40-180 bpm)
    # This gives peak width of ~0.1-0.4 seconds
    min_width = int(0.08 * fs)  # Minimum peak width in samples
    max_width = int(0.35 * fs)  # Maximum peak width in samples

    # Find peaks with constraints
    peak_properties = scipy_signal.find_peaks(
        detrended,
        height=height,
        prominence=prominence,
        distance=int(min_distance_s * fs),
        width=(min_width, max_width),
        rel_height=0.75,  # Half prominence for width calculation
    )

    peaks = peak_properties[0]

    if len(peaks) < 2:
        # Fallback: try with relaxed constraints
        peak_properties = scipy_signal.find_peaks(
            detrended,
            height=median_val + 0.3 * mad,
            prominence=0.2 * mad,
            distance=int(min_distance_s * fs),
        )
        peaks = peak_properties[0]

    # Return both peaks and count for feature extraction
    peak_count = len(peaks)
    if peak_count < 2:
        return np.array([], dtype=int), 0

    return peaks.astype(int), peak_count


def bvp_features(bvp_window: np.ndarray, fs: float = 64.0) -> dict[str, float]:
    """Extract HR and HRV features from a 60-second BVP window.

    HR: heart rate in bpm
    HRV: NN intervals + frequency-domain energy (ULF, LF, HF, UHF)
    """
    features: dict[str, float] = {}
    sig = bvp_window.ravel()

    if len(sig) == 0:
        return {k: 0.0 for k in _BVP_FEATURE_KEYS}

    # Find peaks using improved algorithm
    peaks, peak_count = _find_peaks_improved(sig, fs=fs)

    if len(peaks) < 2:
        # Return zeros but include peak_count for model to assess signal quality
        features = {k: 0.0 for k in _BVP_FEATURE_KEYS}
        features["bvp_peak_count"] = 0.0
        return features

    # HR from inter-peak intervals (in seconds)
    peak_diffs = np.diff(peaks) / fs
    raw_hr = 60.0 / peak_diffs
    plausible = (raw_hr >= 40.0) & (raw_hr <= 200.0)
    peak_diffs = peak_diffs[plausible]
    if len(peak_diffs) < 2:
        features = {k: 0.0 for k in _BVP_FEATURE_KEYS}
        features["bvp_peak_count"] = float(peak_count)
        return features

    hr = 60.0 / peak_diffs
    features["bvp_hr_mean"] = float(np.mean(hr))
    features["bvp_hr_std"] = float(np.std(hr, ddof=1)) if len(hr) > 1 else 0.0

    # NN intervals: use peak intervals directly (in seconds)
    nn_intervals = peak_diffs
    features["bvp_hrv_mean_nn"] = float(np.mean(nn_intervals))
    features["bvp_hrv_std_nn"] = float(np.std(nn_intervals, ddof=1)) if len(nn_intervals) > 1 else 0.0
    features["bvp_hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(nn_intervals) ** 2))) if len(nn_intervals) > 1 else 0.0

    # Peak count feature (signal quality indicator)
    features["bvp_peak_count"] = float(peak_count)

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
            
            # LF/HF ratio (classic stress indicator)
            lf = features["bvp_hrv_lf"]
            hf = features["bvp_hrv_hf"]
            if hf <= 1e-6:
                features["bvp_hrv_lf_hf_ratio"] = 0.0
            else:
                features["bvp_hrv_lf_hf_ratio"] = float(np.clip(lf / hf, 0.0, 1000.0))
        else:
            for band in ["ulf", "lf", "hf", "uhf"]:
                features[f"bvp_hrv_{band}"] = 0.0
            features["bvp_hrv_lf_hf_ratio"] = 0.0
    else:
        for band in ["ulf", "lf", "hf", "uhf"]:
            features[f"bvp_hrv_{band}"] = 0.0
        features["bvp_hrv_lf_hf_ratio"] = 0.0

    # Sample Entropy (non-linear dynamics, strong stress indicator)
    # Sample Entropy measures signal regularity/complexity
    # Lower entropy = more regular/predictable = often associated with stress
    if len(nn_intervals) >= 5:
        features["bvp_hrv_sampen"] = float(_sample_entropy(nn_intervals, m=2, r=0.2))
    else:
        features["bvp_hrv_sampen"] = 0.0

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
    "bvp_hrv_lf_hf_ratio",
    "bvp_hrv_sampen",
    "bvp_peak_count",
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
    features["eda_std"] = float(np.std(sig, ddof=1)) if len(sig) > 1 else 0.0
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
    scl_std = float(np.std(scl, ddof=1)) if len(scl) > 1 else 0.0
    features["eda_scl_mean"] = scl_mean
    features["eda_scl_std"] = scl_std

    # Phasic / SCR component
    scr = sig - scl
    scr_mean = float(np.mean(scr))
    scr_std = float(np.std(scr, ddof=1)) if len(scr) > 1 else 0.0
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
        "temp_std": float(np.std(sig, ddof=1)) if len(sig) > 1 else 0.0,
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
