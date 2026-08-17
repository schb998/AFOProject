"""
Spectrum Analysis for Gait Data
================================
Performs Power Spectral Density (PSD) analysis and Winter's Residual Analysis
on raw marker (TRC) and force plate (MOT) data to recommend optimal filter
type and cutoff frequency.

Usage:
    python spectrum_analysis.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.fft import rfft, rfftfreq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from resources.file_types.trc import TRC
from resources.file_types.mot import MOT


# ─────────────────────────────── CONFIG ────────────────────────────────────
TRC_PATH = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\Gait01\afo speed 0.trc"
MOT_PATH = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\Gait01\afo speed 0.mot"
OUTPUT_DIR = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\Gait01\spectrum_analysis"

# Cutoff range to test in residual analysis
RESIDUAL_CUTOFF_MIN = 1.0    # Hz
RESIDUAL_CUTOFF_MAX_KIN = 30.0   # Hz (markers cap at fs/2 = 50 Hz)
RESIDUAL_CUTOFF_MAX_GRF = 100.0  # Hz (GRF cap at 100 Hz for residual test)

FILTER_ORDER = 4  # Butterworth order to recommend
# ────────────────────────────────────────────────────────────────────────────


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (forward-backward = order*2)."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        return data.copy()
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data, padlen=min(len(data) - 1, 3 * max(len(a), len(b))))


def compute_psd(data: np.ndarray, fs: float):
    """Compute Power Spectral Density using Welch's method."""
    # Use ~5-second windows for good frequency resolution
    nperseg = min(len(data), int(fs * 5))
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, scaling='density')
    return freqs, psd


def winters_residual_analysis(data: np.ndarray, fs: float, cutoffs: np.ndarray, order: int = 4):
    """
    Winter's (1990) Residual Analysis.

    For each cutoff frequency, filters the signal and computes the RMS of the
    residual (original - filtered). The optimal cutoff is at the "elbow" of the
    residual curve — where the residual is no longer dominated by noise.

    Returns:
        residuals: RMS residual at each cutoff
    """
    residuals = []
    for fc in cutoffs:
        filtered = butter_lowpass_filter(data, fc, fs, order)
        residual = data - filtered
        rms = np.sqrt(np.mean(residual ** 2))
        residuals.append(rms)
    return np.array(residuals)


def find_optimal_cutoff(cutoffs: np.ndarray, residuals: np.ndarray) -> float:
    """
    Find the optimal cutoff frequency using the two-line intersection method.

    Fits a steep line to the declining portion and a flat line to the plateau,
    then returns the intersection frequency (the elbow). This is more robust
    than the second-derivative method for gait data with very steep initial drops.
    """
    n = len(cutoffs)
    best_score = np.inf
    best_idx = n // 4  # Start search from 25% of the range

    for split in range(n // 10, n - n // 10):
        # Left segment: steep decline
        c_left = cutoffs[:split + 1]
        r_left = residuals[:split + 1]
        # Right segment: plateau
        c_right = cutoffs[split:]
        r_right = residuals[split:]

        # Fit lines
        if len(c_left) < 2 or len(c_right) < 2:
            continue
        p_left = np.polyfit(c_left, r_left, 1)
        p_right = np.polyfit(c_right, r_right, 1)

        # Residuals from lines
        err = np.sum((r_left - np.polyval(p_left, c_left)) ** 2) + \
              np.sum((r_right - np.polyval(p_right, c_right)) ** 2)

        if err < best_score:
            best_score = err
            best_idx = split

    return float(cutoffs[best_idx])


def cumulative_power_cutoff(freqs: np.ndarray, psd: np.ndarray, threshold: float = 0.99) -> float:
    """Return the frequency that contains `threshold` fraction of total power."""
    cumpower = np.cumsum(psd)
    cumpower /= cumpower[-1]
    idx = np.searchsorted(cumpower, threshold)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


# ════════════════════════════════════════════════════════════════════════════
# KINEMATICS  (TRC markers, 100 Hz)
# ════════════════════════════════════════════════════════════════════════════

def analyze_kinematics(output_dir: str):
    print("\n" + "=" * 60)
    print("KINEMATICS  (Marker Trajectories — TRC, 100 Hz)")
    print("=" * 60)

    trc = TRC.load_from_trc(TRC_PATH)
    fs = float(trc.metadata.camera_rate)
    print(f"  Sampling rate: {fs:.0f} Hz | Markers: {trc.metadata.num_markers} | Frames: {trc.metadata.num_frames}")

    # Collect all non-NaN marker coordinate signals
    data_cols = [c for c in trc.data.columns if c != 'Time']
    signals = []
    for col in data_cols:
        sig = trc.data[col].values.astype(float)
        if not np.all(np.isnan(sig)):
            sig = sig.copy()
            # Fill NaN by linear interpolation
            nans = np.isnan(sig)
            if nans.any() and (~nans).sum() > 2:
                sig[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], sig[~nans])
            signals.append(sig)

    print(f"  Using {len(signals)} non-empty marker channels for analysis")

    # ── Average PSD across all channels ──────────────────────────────────
    all_psds = []
    for sig in signals:
        f, p = compute_psd(sig, fs)
        all_psds.append(p)
    mean_psd = np.mean(all_psds, axis=0)
    freqs = f

    p99 = cumulative_power_cutoff(freqs, mean_psd, threshold=0.99)
    p999 = cumulative_power_cutoff(freqs, mean_psd, threshold=0.999)
    print(f"  99%  of cumulative power below: {p99:.1f} Hz")
    print(f"  99.9% of cumulative power below: {p999:.1f} Hz")

    # ── Winter's Residual Analysis on a representative signal ─────────────
    # Use the marker with highest variance (most informative)
    variances = [np.nanvar(s) for s in signals]
    best_sig = signals[int(np.argmax(variances))]
    best_col = data_cols[int(np.argmax(variances))]

    cutoffs_kin = np.linspace(RESIDUAL_CUTOFF_MIN, RESIDUAL_CUTOFF_MAX_KIN, 120)
    residuals_kin = winters_residual_analysis(best_sig, fs, cutoffs_kin, order=FILTER_ORDER)
    optimal_fc_kin = find_optimal_cutoff(cutoffs_kin, residuals_kin)
    print(f"  Winter's residual analysis optimal cutoff: {optimal_fc_kin:.1f} Hz  (channel: {best_col})")

    # ── Filter comparison ─────────────────────────────────────────────────
    fc_to_compare = [6, 10, 15, 20]  # Hz options
    t = trc.data['Time'].values[:500]   # First 5 seconds for visibility
    seg = best_sig[:500]

    # ─────────────── FIGURE 1: PSD ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Kinematics (Marker Trajectories) — Spectrum Analysis", fontsize=15, fontweight='bold')

    ax = axes[0, 0]
    ax.semilogy(freqs, mean_psd, color='steelblue', alpha=0.9, linewidth=1.5)
    ax.axvline(p99, color='orangered', linestyle='--', label=f'99% power: {p99:.1f} Hz')
    ax.axvline(p999, color='crimson', linestyle=':', label=f'99.9% power: {p999:.1f} Hz')
    ax.axvline(optimal_fc_kin, color='limegreen', linestyle='-', linewidth=2,
               label=f'Winter optimal: {optimal_fc_kin:.1f} Hz')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (mm²/Hz, log scale)")
    ax.set_title("Average Power Spectral Density (all markers)")
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([0, fs / 2])

    ax = axes[0, 1]
    cumpower = np.cumsum(mean_psd) / np.sum(mean_psd)
    ax.plot(freqs, cumpower * 100, color='steelblue', linewidth=2)
    ax.axvline(p99, color='orangered', linestyle='--', label=f'99%: {p99:.1f} Hz')
    ax.axvline(p999, color='crimson', linestyle=':', label=f'99.9%: {p999:.1f} Hz')
    ax.axhline(99, color='orangered', linestyle='--', alpha=0.3)
    ax.axhline(99.9, color='crimson', linestyle=':', alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cumulative Power (%)")
    ax.set_title("Cumulative Power vs Frequency")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, fs / 2])
    ax.set_ylim([0, 101])

    ax = axes[1, 0]
    ax.plot(cutoffs_kin, residuals_kin, 'steelblue', linewidth=2)
    ax.axvline(optimal_fc_kin, color='limegreen', linewidth=2, linestyle='--',
               label=f'Optimal: {optimal_fc_kin:.1f} Hz')
    for fc in fc_to_compare:
        ax.axvline(fc, color='gray', linewidth=1, linestyle=':', alpha=0.7)
        ax.text(fc + 0.2, residuals_kin.max() * 0.95, f'{fc} Hz', fontsize=8, color='gray', va='top')
    ax.set_xlabel("Cutoff Frequency (Hz)")
    ax.set_ylabel("RMS Residual (mm)")
    ax.set_title(f"Winter's Residual Analysis\n(channel: {best_col})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, seg, 'k', linewidth=1.2, alpha=0.5, label='Raw')
    colors_compare = ['steelblue', 'orangered', 'limegreen', 'purple']
    for fc, col in zip(fc_to_compare, colors_compare):
        filt = butter_lowpass_filter(seg, fc, fs, FILTER_ORDER)
        ax.plot(t, filt, color=col, linewidth=1.5, label=f'{fc} Hz Butterworth')
    ax.axvline(t[0] + 0.5, color='gray', alpha=0)  # Spacing hack
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (mm)")
    ax.set_title(f"Filtered vs Raw — {best_col}\n(Butterworth order {FILTER_ORDER}, zero-phase)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "kinematics_spectrum_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    return optimal_fc_kin, p99


# ════════════════════════════════════════════════════════════════════════════
# KINETICS  (Ground Reaction Forces — MOT, 2000 Hz)
# ════════════════════════════════════════════════════════════════════════════

def analyze_kinetics(output_dir: str):
    print("\n" + "=" * 60)
    print("KINETICS  (Ground Reaction Forces — MOT, 2000 Hz)")
    print("=" * 60)

    mot = MOT.load_from_mot(MOT_PATH)
    time = mot.data['time'].values
    fs = 1.0 / np.mean(np.diff(time))
    print(f"  Sampling rate: {fs:.0f} Hz")

    # Analyse all three vertical + horizontal force channels
    channels = {
        'Right Fy (vert)': 'ground_force2_vy',
        'Left  Fy (vert)': 'ground_force1_vy',
        'Right Fx (horiz)': 'ground_force2_vx',
        'Left  Fx (horiz)': 'ground_force1_vx',
        'Right Fz (a-p)': 'ground_force2_vz',
        'Left  Fz (a-p)': 'ground_force1_vz',
    }
    channels = {k: v for k, v in channels.items() if v in mot.data.columns}

    # ── Per-channel PSD ───────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Kinetics (GRF) — Power Spectral Density", fontsize=15, fontweight='bold')

    cutoffs_grf = np.linspace(RESIDUAL_CUTOFF_MIN, RESIDUAL_CUTOFF_MAX_GRF, 200)
    optimal_fcs = {}
    p99s = {}

    for idx, (label, col) in enumerate(channels.items()):
        row, colnum = divmod(idx, 2)
        ax = axes[row, colnum]

        sig = mot.data[col].values.astype(float)
        freqs, psd = compute_psd(sig, fs)

        p99 = cumulative_power_cutoff(freqs, psd, 0.99)
        p999 = cumulative_power_cutoff(freqs, psd, 0.999)
        p99s[label] = p99

        ax.semilogy(freqs, psd, linewidth=1, alpha=0.85)
        ax.axvline(p99, color='orangered', linestyle='--', linewidth=1.5, label=f'99%: {p99:.0f} Hz')
        ax.axvline(p999, color='crimson', linestyle=':', linewidth=1.5, label=f'99.9%: {p999:.0f} Hz')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (N²/Hz, log)")
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim([0, 200])  # Show up to 200 Hz

    plt.tight_layout()
    psd_path = os.path.join(output_dir, "kinetics_psd.png")
    plt.savefig(psd_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PSD figure saved: {psd_path}")

    # ── Winter's Residual Analysis on vertical forces ─────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
    fig2.suptitle("Kinetics (GRF) — Residual Analysis & Filter Comparison", fontsize=15, fontweight='bold')

    # Vert force channels only for residual analysis
    vert_channels = {k: v for k, v in channels.items() if 'vert' in k.lower() or 'Fy' in k}
    fc_to_compare_grf = [10, 20, 50, 100]
    colors_grf = ['steelblue', 'orangered', 'limegreen', 'purple']

    ax_res_r = axes2[0, 0]
    ax_res_l = axes2[0, 1]
    ax_filt_r = axes2[1, 0]
    ax_filt_l = axes2[1, 1]

    seg_t = time[:2000]  # 1 second

    for label, col, ax_res, ax_filt in [
        ('Right Fy (vert)', 'ground_force2_vy', ax_res_r, ax_filt_r),
        ('Left  Fy (vert)', 'ground_force1_vy', ax_res_l, ax_filt_l),
    ]:
        if col not in mot.data.columns:
            continue
        sig = mot.data[col].values.astype(float)
        residuals_grf = winters_residual_analysis(sig, fs, cutoffs_grf, order=FILTER_ORDER)
        opt_fc = find_optimal_cutoff(cutoffs_grf, residuals_grf)
        optimal_fcs[label] = opt_fc
        print(f"  Winter's optimal cutoff — {label}: {opt_fc:.1f} Hz")

        ax_res.plot(cutoffs_grf, residuals_grf, linewidth=2, label='RMS Residual')
        ax_res.axvline(opt_fc, color='limegreen', linewidth=2, linestyle='--',
                       label=f'Optimal: {opt_fc:.1f} Hz')
        for fc in fc_to_compare_grf:
            ax_res.axvline(fc, color='gray', linewidth=1, linestyle=':', alpha=0.7)
            ax_res.text(fc + 0.5, residuals_grf.max() * 0.92, f'{fc}', fontsize=8, color='gray', va='top')
        ax_res.set_xlabel("Cutoff Frequency (Hz)")
        ax_res.set_ylabel("RMS Residual (N)")
        ax_res.set_title(f"Winter's Residual — {label.strip()}")
        ax_res.legend(fontsize=9)
        ax_res.grid(True, alpha=0.3)

        seg = sig[:2000]
        ax_filt.plot(seg_t, seg, 'k', linewidth=0.8, alpha=0.5, label='Raw')
        for fc, col_c in zip(fc_to_compare_grf, colors_grf):
            filt = butter_lowpass_filter(seg, fc, fs, FILTER_ORDER)
            ax_filt.plot(seg_t, filt, color=col_c, linewidth=1.5, label=f'{fc} Hz')
        ax_filt.set_xlabel("Time (s)")
        ax_filt.set_ylabel("Force (N)")
        ax_filt.set_title(f"Filter Comparison — {label.strip()}\n(Butterworth order {FILTER_ORDER}, zero-phase)")
        ax_filt.legend(fontsize=9)
        ax_filt.grid(True, alpha=0.3)

    plt.tight_layout()
    res_path = os.path.join(output_dir, "kinetics_residual_filter.png")
    plt.savefig(res_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Residual/filter figure saved: {res_path}")

    return optimal_fcs, p99s


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════

def print_recommendations(kin_fc, kin_p99, grf_fcs, grf_p99s):
    sep = "=" * 64
    print("\n" + sep)
    print("  FILTER RECOMMENDATIONS SUMMARY")
    print(sep)
    print("  KINEMATICS (Marker Trajectories, 100 Hz)")
    print(f"    99%  of cumulative power below : {kin_p99:.1f} Hz")
    print(f"    Winter's residual optimal cutoff: {kin_fc:.1f} Hz")
    rec_kin = max(4.0, min(kin_fc, 15.0))
    print(f"    >> RECOMMENDED: {FILTER_ORDER}th-order Butterworth (zero-phase)")
    print(f"       Cutoff = {rec_kin:.0f} Hz")
    print(sep)
    print("  KINETICS (GRFs, 2000 Hz)")
    for label, fc in grf_fcs.items():
        p99 = grf_p99s.get(label, 0)
        print(f"    {label.strip():<30} 99% power: {p99:5.0f} Hz  | Winter optimal: {fc:.1f} Hz")
    rec_grf = max(15.0, min(np.mean(list(grf_fcs.values())), 60.0))
    print(f"    >> RECOMMENDED: {FILTER_ORDER}th-order Butterworth (zero-phase)")
    print(f"       Cutoff = {rec_grf:.0f} Hz")
    print(sep)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    kin_fc, kin_p99 = analyze_kinematics(OUTPUT_DIR)
    grf_fcs, grf_p99s = analyze_kinetics(OUTPUT_DIR)
    print_recommendations(kin_fc, kin_p99, grf_fcs, grf_p99s)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
