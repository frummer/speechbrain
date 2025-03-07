import numpy as np

def is_silent(audio_window: np.ndarray, 
              silence_rms_threshold: float = 1e-4, 
              mean_amplitude_threshold: float = 1e-4) -> bool:
    """
    Returns True if the RMS of audio_window is below the given threshold
    and if the mean amplitude is also below mean_amplitude_threshold.
    """
    rms = np.sqrt(np.mean(audio_window**2))
    avg_amplitude = np.mean(np.abs(audio_window))
    return rms < silence_rms_threshold and avg_amplitude < mean_amplitude_threshold


def compute_leakage_for_pair(
    data1: np.ndarray, 
    data2: np.ndarray, 
    window_size: int,
    stage,
    silence_rms_threshold: float = 1e-4, 
    mean_amplitude_threshold: float = 1e-4,

):
    """
    Computes the correlation over sliding windows between data1 and data2.
    Window step is half of window_size. Skips silent windows (marked as NaN).
    Returns an array of correlations (one per valid window).
    """
    window_step = window_size // 2
    n_samples = len(data1)

    # Edge case: if not enough samples, return empty:
    if n_samples < window_size:
        return []

    # Window start indices
    win_indices = range(0, n_samples - window_size + 1, window_step)

    correlations = []
    for start in win_indices:
        window1 = data1[start : start + window_size]
        window2 = data2[start : start + window_size]

        # Skip if silent
        if (is_silent(window1, silence_rms_threshold, mean_amplitude_threshold) or
            is_silent(window2, silence_rms_threshold, mean_amplitude_threshold)):
            continue

        # Also skip if zero variance in either window
        if np.std(window1) == 0 or np.std(window2) == 0:
            continue

        corr = np.corrcoef(window1, window2)[0, 1]
        correlations.append(corr)

    return correlations