"""Load Sensor Logger recordings, window them, and extract HAR features."""
import os
import numpy as np
import pandas as pd

ACTIVITIES = ["still", "standing", "walking", "jumping"]
SAMPLE_RATE_HZ = 100.0  # confirmed from Metadata.csv (sampleRateMs=10) across all sessions


def list_sessions(data_dir):
    """{activity: [session_dir, ...]} for every recording under data_dir/<activity>/."""
    sessions = {a: [] for a in ACTIVITIES}
    for activity in ACTIVITIES:
        act_dir = os.path.join(data_dir, activity)
        for name in sorted(os.listdir(act_dir)):
            session_dir = os.path.join(act_dir, name)
            if os.path.isdir(session_dir):
                sessions[activity].append(session_dir)
    return sessions


def load_session(session_dir):
    """Load one session's Accelerometer.csv and Gyroscope.csv and merge them by row
    position (both streams share identical timestamps at 100 Hz), plus derived
    3D magnitudes a_mag/g_mag."""
    acc = pd.read_csv(os.path.join(session_dir, "Accelerometer.csv"))
    gyro = pd.read_csv(os.path.join(session_dir, "Gyroscope.csv"))
    n = min(len(acc), len(gyro))
    acc, gyro = acc.iloc[:n], gyro.iloc[:n]
    df = pd.DataFrame({
        "t": acc["seconds_elapsed"].values,
        "ax": acc["x"].values, "ay": acc["y"].values, "az": acc["z"].values,
        "gx": gyro["x"].values, "gy": gyro["y"].values, "gz": gyro["z"].values,
    })
    df["a_mag"] = np.sqrt(df.ax**2 + df.ay**2 + df.az**2)
    df["g_mag"] = np.sqrt(df.gx**2 + df.gy**2 + df.gz**2)
    return df


def _dominant_freq_and_energy(signal, fs):
    """Dominant frequency (excluding DC), spectral energy, and spectral entropy from
    the FFT of one 1D signal."""
    n = len(signal)
    sig = signal - signal.mean()
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(spec) ** 2
    power[0] = 0.0  # drop DC
    dom_freq = freqs[np.argmax(power)] if power.sum() > 0 else 0.0
    energy = power.sum() / n
    p = power / power.sum() if power.sum() > 0 else np.ones_like(power) / len(power)
    entropy = -np.sum(p * np.log2(p + 1e-12))
    return dom_freq, energy, entropy


def extract_window_features(win, fs=SAMPLE_RATE_HZ):
    """The 32 time- and frequency-domain features for one window: per-axis mean/std,
    SMA, and inter-axis correlations for accel and gyro, plus magnitude mean/std/RMS
    and FFT dominant frequency/energy/entropy for both sensors."""
    feats = {}
    for axis_set, prefix in [(["ax", "ay", "az"], "acc"), (["gx", "gy", "gz"], "gyro")]:
        x, y, z = (win[c].values for c in axis_set)
        feats[f"{prefix}_mean_x"] = x.mean()
        feats[f"{prefix}_mean_y"] = y.mean()
        feats[f"{prefix}_mean_z"] = z.mean()
        feats[f"{prefix}_std_x"] = x.std()
        feats[f"{prefix}_std_y"] = y.std()
        feats[f"{prefix}_std_z"] = z.std()
        feats[f"{prefix}_sma"] = (np.abs(x) + np.abs(y) + np.abs(z)).mean()
        feats[f"{prefix}_corr_xy"] = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0
        feats[f"{prefix}_corr_yz"] = np.corrcoef(y, z)[0, 1] if y.std() > 0 and z.std() > 0 else 0.0
        feats[f"{prefix}_corr_xz"] = np.corrcoef(x, z)[0, 1] if x.std() > 0 and z.std() > 0 else 0.0

    for mag_col, prefix in [("a_mag", "acc"), ("g_mag", "gyro")]:
        mag = win[mag_col].values
        feats[f"{prefix}_mag_mean"] = mag.mean()
        feats[f"{prefix}_mag_std"] = mag.std()
        feats[f"{prefix}_mag_rms"] = np.sqrt((mag ** 2).mean())
        dom_freq, energy, entropy = _dominant_freq_and_energy(mag, fs)
        feats[f"{prefix}_dom_freq"] = dom_freq
        feats[f"{prefix}_spec_energy"] = energy
        feats[f"{prefix}_spec_entropy"] = entropy
    return feats


def window_session(df, window_size, step):
    """(start, end) row-index pairs slicing df into window_size-row windows, step
    rows apart. Falls back to one window covering the whole session if it's shorter
    than window_size."""
    n = len(df)
    windows = []
    start = 0
    while start + window_size <= n:
        windows.append((start, start + window_size))
        start += step
    if not windows:
        windows = [(0, n)]
    return windows


def build_feature_table(data_dir, window_sec=1.0, overlap=0.5, fs=SAMPLE_RATE_HZ):
    """Window every session under data_dir and extract features for each window.
    window_sec/overlap control window length and step, both converted to samples
    from fs so they're always an exact multiple of the sampling rate. Returns the
    feature table (one row per window), the list of feature column names, and the
    window size/step actually used in samples."""
    window_size = int(round(window_sec * fs))
    step = max(1, int(round(window_size * (1 - overlap))))
    sessions = list_sessions(data_dir)
    rows = []
    for activity, session_dirs in sessions.items():
        for session_dir in session_dirs:
            session_id = os.path.basename(session_dir)
            df = load_session(session_dir)
            for w_idx, (s, e) in enumerate(window_session(df, window_size, step)):
                win = df.iloc[s:e]
                feats = extract_window_features(win)
                feats.update({
                    "activity": activity,
                    "session_id": session_id,
                    "window_idx": w_idx,
                    "t_start": win["t"].iloc[0],
                })
                rows.append(feats)
    table = pd.DataFrame(rows)
    meta_cols = ["activity", "session_id", "window_idx", "t_start"]
    feature_cols = [c for c in table.columns if c not in meta_cols]
    return table, feature_cols, window_size, step
