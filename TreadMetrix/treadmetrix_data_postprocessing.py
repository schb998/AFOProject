from resources.filetypes_gestion.mot import MOT
import local_paths as local

import os
from copy import deepcopy
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import numpy as np

def filter_grf(mot, fs):
    """Filters data of a MOT object with a Butterworth filter.

    Args:
        mot  (MOT): MOT object whose data is to be filtered.
        fs (float): sampling frequency.
    """
    b, a        = butter(6, 12 / (fs / 2), btype='low')
    filtered_df = deepcopy(mot.data)
    for col in mot.data.columns.tolist():
        filtered_df[col] = filtfilt(b, a, mot.data[col])
    mot.data = filtered_df
    return None

def baseline_correct_debug(mot, fz_col, related_cols, output_path, show=False):
    """Corrects the baseline of one of the columns of the m data.

    Saves plot of the baseline correction.

    Args:
        mot                     (MOT): data to process.
        fz_col               (string): name of the column to correct.
        related_cols (list of string): other columns to consider.
        output_path          (string): output path for plot save.
        show                   (bool): whether to show the figure when method is called.
    """
    fy                = mot.data[fz_col]
    corrected_df      = deepcopy(mot.data)
    valley_indices, _ = find_peaks(-fy)
    swing_valleys     = valley_indices[fy[valley_indices] < 0]

    print(f"\nCorrecting {fz_col}")
    print(f"Number of swing valleys below 0N: {len(swing_valleys)}")

    time_scale = mot.data['time'] if 'time' in mot.data.columns.tolist() else np.arange(len(fy))
    plt.figure(figsize=(12, 4))
    plt.plot(time_scale, fy, label='Original', alpha=0.7)
    plt.scatter(time_scale[swing_valleys], fy[swing_valleys], color='red', label='Swing Valleys')

    if len(swing_valleys) == 0:
        print("No valleys found below zero. Skipping correction.")
        return corrected_df

    baseline = abs(np.median(fy[swing_valleys]))
    print(f"Baseline offset to add: {baseline:.2f}")
    corrected_df[fz_col] = fy + baseline

    for col in related_cols:
        related           = mot.data[col]
        offset            = np.median(related[swing_valleys])
        corrected_df[col] = related - offset if offset > 0 else related + abs(offset)
        print(f"Offset for {col}: {offset:.2f}")

    plt.plot(time_scale, corrected_df[fz_col], label='Corrected', alpha=0.8)
    plt.title(f"{fz_col} Baseline Correction")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"{ mot.filename.replace('.m', '') }_baseline_correction_{fz_col}.png"),
                bbox_inches='tight')
    mot.data = corrected_df
    if show:
        plt.show()
    return None

def detect_toe_offs(zeroed_mot, fs, threshold=20):
    """Detects the gait cycles' "toe off" points.

    Args:
        zeroed_mot (MOT): data to process.
        fs       (float): sampling frequency.
        threshold  (int): threshold to use for detection.

    Returns:
        Dictionary of the data's toe offs, listed by side.

    """
    toe_offs = {'R': [], 'L': []}

    if 'ground_force2_vy' in zeroed_mot.data.columns.tolist():
        rzf          = zeroed_mot.data['ground_force2_vy']
        r_indexes, _ = find_peaks(rzf, prominence=15, distance=int(fs / 10), height=200)
        peak = 0
        while peak < len(r_indexes):
            idx_start    = r_indexes[peak]
            below_thresh = np.where(rzf[idx_start:] < threshold)[0]
            if len(below_thresh) == 0:
                break
            toe_offs['R'].append(idx_start + below_thresh[0])
            next_peaks = np.where(r_indexes > toe_offs['R'][-1])[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]

    if 'ground_force1_vy' in zeroed_mot.data.columns.tolist():
        lzf          = zeroed_mot.data['ground_force1_vy']
        l_indexes, _ = find_peaks(lzf, prominence=15, distance=int(fs / 10), height=200)
        peak = 0
        while peak < len(l_indexes):
            idx_start    = l_indexes[peak]
            below_thresh = np.where(lzf[idx_start:] < threshold)[0]
            if len(below_thresh) == 0:
                break
            toe_offs['L'].append(idx_start + below_thresh[0])
            next_peaks = np.where(l_indexes > toe_offs['L'][-1])[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]

    return toe_offs

def detect_heel_strikes(zeroed_mot, fs, threshold=20):
    """Detects the gait cycles' "heel strikes" points.
    Args:
        zeroed_mot (MOT): data to process.
        fs       (float): sampling frequency.
        threshold  (int): threshold to use for detection.

    Returns:
        Dictionary of the data's heel strikes, listed by side.

    """
    heel_contacts = {'R': [], 'L': []}
    distance      = int(fs / 2)

    if 'ground_force2_vy' in zeroed_mot.data.columns.tolist():
        rzf          = zeroed_mot.data['ground_force2_vy']
        r_indexes, _ = find_peaks(-rzf, prominence=14, distance=distance, height=-100)
        rest         = len(rzf)
        peak         = 0
        while peak < len(r_indexes) and rest > 1000:
            idx_start    = r_indexes[peak]
            above_thresh = np.where(rzf[idx_start:] > threshold)[0]
            if len(above_thresh) == 0:
                break
            heel_idx   = idx_start + above_thresh[0]
            heel_contacts['R'].append(heel_idx)
            next_peaks = np.where(r_indexes > heel_idx)[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]
            rest = len(rzf[r_indexes[peak]:])

    if 'ground_force1_vy' in zeroed_mot.data.columns.tolist():
        lzf          = zeroed_mot.data['ground_force1_vy']
        l_indexes, _ = find_peaks(-lzf, prominence=14, distance=distance, height=-100)
        rest         = len(lzf)
        peak         = 0
        while peak < len(l_indexes) and rest > int(fs / 2):
            idx_start    = l_indexes[peak]
            above_thresh = np.where(lzf[idx_start:] > threshold)[0]
            if len(above_thresh) == 0:
                break
            heel_idx   = idx_start + above_thresh[0]
            heel_contacts['L'].append(heel_idx)
            next_peaks = np.where(l_indexes > heel_idx)[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]
            rest = len(lzf[l_indexes[peak]:])

    return heel_contacts

def zero_swing_phase(mot_df, toe_offs, heel_strikes, side):
    """Sets GRF and related columns to zero between toe-off and next heel strike.

    Args:
        mot_df       (MOT): data to process.
        toe_offs     (dict): toe off moments.
        heel_strikes (dict): heel strikes moment.
        side         (string): side to focus on. Method raises ValueError if invalid.

    Raises:
        ValueError: if the 'side' argument is not either 'R'/'r'/'right' or 'L'/'l'/'left'.
    """
    df_corrected = deepcopy(mot_df.data)

    side = side.lower()

    if side == 'r' or side == 'right':
        to_list = toe_offs['R']
        hs_list = heel_strikes['R']
        cols_to_zero = ['ground_force2_vx', 'ground_force2_vy', 'ground_force2_vz',
                        'ground_force2_px', 'ground_force2_py', 'ground_force2_pz',
                        'ground_torque2_x', 'ground_torque2_y', 'ground_torque2_z']
    elif side == 'l' or side == 'left':
        to_list = toe_offs['L']
        hs_list = heel_strikes['L']
        cols_to_zero = ['ground_force1_vx', 'ground_force1_vy', 'ground_force1_vz',
                        'ground_force1_px', 'ground_force1_py', 'ground_force1_pz',
                        'ground_torque1_x', 'ground_torque1_y', 'ground_torque1_z']
    else:
        raise ValueError("Side must be 'R' or 'L'")

    for toe_idx in to_list:
        hs_after_toe = [hs for hs in hs_list if hs > toe_idx]
        if hs_after_toe:
            heel_idx = hs_after_toe[0]
            for col in cols_to_zero:
                if col in df_corrected.columns.tolist():
                    df_corrected.loc[toe_idx:heel_idx, col] = 0

    mot_df.data = df_corrected
    return None

def plot_grf_details(mot, heel_strikes, toe_offs, show=False):
    """Saves plot of the vertical forces with toe offs and heel strikes.

    Args:
        mot           (MOT): MOT object of the data.
        heel_strikes (dict): heel strikes moment.
        toe_offs     (dict): toe offs moment.
        show         (bool): whether to show the figure when method is called.

    """
    plt.figure(figsize=(14, 6))
    right_fy = mot.data['ground_force2_vy']
    left_fy  = mot.data['ground_force1_vy']
    time     = mot.data['time']

    plt.plot(time, right_fy, label='Right Fy', alpha=0.7)
    plt.plot(time, left_fy, label='Left Fy', alpha=0.7)

    # Toe-offs
    plt.scatter([time[i] for i in toe_offs['R']], [right_fy[i] for i in toe_offs['R']],
                color='red', marker='x', label='Right Toe-Offs')
    plt.scatter([time[i] for i in toe_offs['L']], [left_fy[i] for i in toe_offs['L']],
                color='green', marker='x', label='Left Toe-Offs')

    # Heel strikes
    plt.scatter([time[i] for i in heel_strikes['R']], [right_fy[i] for i in heel_strikes['R']],
                color='blue', marker='o', label='Right Heel Strikes')
    plt.scatter([time[i] for i in heel_strikes['L']], [left_fy[i] for i in heel_strikes['L']],
                color='purple', marker='o', label='Left Heel Strikes')

    plt.title(f"Vertical GRFs with Toe-Offs and Heel Strikes: {mot.filename}")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output,
                             f"{mot.filename.replace('.m', '')}_vertical_grfs_with_toeoffs_heelstrikes.png"),
                bbox_inches='tight')
    if show:
        plt.show()
    return None

if __name__ == "__main__":
    raw_data_path = local.get_raw_mot_path()
    output        = local.get_corrected_mot_path()
    os.makedirs(output, exist_ok=True)
    file_list = sorted(f for f in os.listdir(raw_data_path) if f.endswith('.mot'))
    files     = [file for file in file_list if not "static" in file.lower()]
    mots = []
    for file in files:
        mots.append(MOT.load(raw_data_path, file))
    for m in mots:
        time       = m.data['time']
        frame_rate = 1 / np.mean(np.diff(time))
        print(f"\nProcessing: {m.filename} with sampling frequency: {frame_rate:.2f} Hz.")
        filter_grf(m, frame_rate)
        baseline_correct_debug(m, 'ground_force2_vy', ['ground_force2_vx', 'ground_force2_vz'], output)
        baseline_correct_debug(m, 'ground_force1_vy', ['ground_force1_vx', 'ground_force1_vz'], output)
        toe_off_moments     = detect_toe_offs(m, frame_rate)
        heel_strike_moments = detect_heel_strikes(m, frame_rate)
        zero_swing_phase(m, toe_off_moments, heel_strike_moments, 'right')
        zero_swing_phase(m, toe_off_moments, heel_strike_moments, 'left')
        m.rename(name =m.filename.replace('.mot', '') + "-corrected",
                 filename = m.filename.replace('.mot', '_corrected.mot'), )
        plot_grf_details(m, heel_strike_moments, toe_off_moments)
        m.save(output)
    print("All files were processed.")