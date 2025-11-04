import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import TreadMetrix.local_paths as local
from resources.filetypes_gestion.mot import MOT

"""
This file is used to process the raw data from MOT files.
    Input: raw MOT files.
    Output: processed MOT file.
"""


def read_mot_files(data_path_mot):
    """
    This function recursively reads data from MOT files.
    Args:
        data_path_mot (string): path to the MOT files' folder.

    Returns:
        list of the files' data, each in the form of a dictionary with keys "filename", "headerLines", "rawData" and "colNames".
    """
    motion_data_list = []
    file_list = sorted(f for f in os.listdir(data_path_mot) if f.endswith('.mot'))

    for filename in file_list:
        file_path = os.path.join(data_path_mot, filename)
        print(f"Reading file: {file_path}")
        try:
            with open(file_path, 'r') as file:
                header_lines = []
                line = next(file).strip("\n")
                while line != "endheader":
                    header_lines.append(line)
                    line = next(file).strip("\n")
                line = next(file)
                while line == "":
                    line = next(file)
                data = pd.read_csv(file, sep=r'\s+')
            motion_data_list.append({
                'fileName': filename,
                'headerLines': header_lines,
                'rawData': data.to_numpy(),
                'colNames': data.columns.to_list()
            })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return motion_data_list


def filter_grf(force_df, fs):
    """
    This function filter data with a Butterworth filter.
    Args:
        force_df (array): data
        fs (float): sampling frequency

    Returns:
        array of the filtered data, in the same shape as the given data.

    """
    b, a = butter(6, 12 / (fs / 2), btype='low')
    filtered_df = force_df.copy()
    for col in force_df.columns:
        filtered_df[col] = filtfilt(b, a, force_df[col].values)
    return filtered_df


def baseline_correct_debug(filtered_df, fz_col, related_cols):
    """
    This function is used to correct the baseline.
    Args:
        filtered_df (dict): data to process
        fz_col (string): column name
        related_cols (list of string): other columns to process

    Returns:
        dict of the processed data, in the same shape as the given data.

    """
    fy = filtered_df[fz_col].values
    corrected_df = filtered_df.copy()

    valley_indices, _ = find_peaks(-fy)
    swing_valleys = valley_indices[fy[valley_indices] < 0]

    print(f"\nCorrecting {fz_col}")
    print(f"Number of swing valleys below 0N: {len(swing_valleys)}")

    """
    time = filtered_df['time'].values if 'time' in filtered_df.columns else np.arange(len(fy))
    plt.figure(figsize=(12, 4))
    plt.plot(time, fy, label='Original', alpha=0.7)
    plt.scatter(time[swing_valleys], fy[swing_valleys], color='red', label='Swing Valleys')
    """

    if len(swing_valleys) == 0:
        print("No valleys found below zero. Skipping correction.")
        return corrected_df

    baseline = abs(np.median(fy[swing_valleys]))
    print(f"Baseline offset to add: {baseline:.2f}")
    corrected_df[fz_col] = fy + baseline

    for col in related_cols:
        related = filtered_df[col].values
        offset = np.median(related[swing_valleys])
        corrected_df[col] = related - offset if offset > 0 else related + abs(offset)
        print(f"Offset for {col}: {offset:.2f}")

    """
    plt.plot(time, corrected_df[fz_col], label='Corrected', alpha=0.8)
    plt.title(f"{fz_col} Baseline Correction")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"{file['fileName']}_baseline_correction_{fz_col}.png"),
                bbox_inches='tight')
    plt.show()
    """

    return corrected_df


def detect_toe_off_from_df(zeroed_df, fs, threshold=20):
    """
    This function detects the gait cycles' "toe off" points.
    Args:
        zeroed_df (dict): data to process
        fs (float): sampling frequency
        threshold (int): threshold to use for detection

    Returns:
        dictionary of the data's toe offs frames, listed by side.

    """
    toe_offs = {'R': [], 'L': []}

    if 'ground_force2_vy' in zeroed_df.columns:
        RZF = zeroed_df['ground_force2_vy'].values
        RIndexes, _ = find_peaks(RZF, prominence=15, distance=int(fs / 10), height=200)
        peak = 0
        while peak < len(RIndexes):
            idx_start = RIndexes[peak]
            below_thresh = np.where(RZF[idx_start:] < threshold)[0]
            if len(below_thresh) == 0:
                break
            toe_offs['R'].append(idx_start + below_thresh[0])
            next_peaks = np.where(RIndexes > toe_offs['R'][-1])[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]

    if 'ground_force1_vy' in zeroed_df.columns:
        LZF = zeroed_df['ground_force1_vy'].values
        LIndexes, _ = find_peaks(LZF, prominence=15, distance=int(fs / 10), height=200)
        peak = 0
        while peak < len(LIndexes):
            idx_start = LIndexes[peak]
            below_thresh = np.where(LZF[idx_start:] < threshold)[0]
            if len(below_thresh) == 0:
                break
            toe_offs['L'].append(idx_start + below_thresh[0])
            next_peaks = np.where(LIndexes > toe_offs['L'][-1])[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]

    return toe_offs


def detect_heel_strikes_from_df(zeroed_df, fs, threshold=20):
    """
    This function detects the gait cycles' "heel strikes" points.
    Args:
        zeroed_df (dict): data to process
        fs (float): sampling frequency
        threshold (int): threshold to use for detection

    Returns:
        dictionary of the data's heel strikes frames, listed by side.

    """
    heel_contacts = {'R': [], 'L': []}
    distance = int(fs / 2)

    if 'ground_force2_vy' in zeroed_df.columns:
        RZF = zeroed_df['ground_force2_vy'].values
        RZF_inv = -RZF
        R_indexes, _ = find_peaks(RZF_inv, prominence=14, distance=distance, height=-100)
        RZF = -RZF_inv
        rest = len(RZF)
        peak = 0
        while peak < len(R_indexes) and rest > 1000:
            idx_start = R_indexes[peak]
            above_thresh = np.where(RZF[idx_start:] > threshold)[0]
            if len(above_thresh) == 0:
                break
            heel_idx = idx_start + above_thresh[0]
            heel_contacts['R'].append(heel_idx)
            next_peaks = np.where(R_indexes > heel_idx)[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]
            rest = len(RZF[R_indexes[peak]:])

    if 'ground_force1_vy' in zeroed_df.columns:
        LZF = zeroed_df['ground_force1_vy'].values
        LZF_inv = -LZF
        L_indexes, _ = find_peaks(LZF_inv, prominence=14, distance=distance, height=-100)
        LZF = -LZF_inv
        rest = len(LZF)
        peak = 0
        while peak < len(L_indexes) and rest > int(fs / 2):
            idx_start = L_indexes[peak]
            above_thresh = np.where(LZF[idx_start:] > threshold)[0]
            if len(above_thresh) == 0:
                break
            heel_idx = idx_start + above_thresh[0]
            heel_contacts['L'].append(heel_idx)
            next_peaks = np.where(L_indexes > heel_idx)[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]
            rest = len(LZF[L_indexes[peak]:])

    return heel_contacts


def zero_swing_phase(df, toe_offs, heel_strikes, side):
    """
    This function sets GRF and related columns to zero between toe-off and next heel strike.
    Raises a ValueError if the 'side' argument is not either 'R' or 'L'
    Args:
        df (dict): data to process
        toe_offs (dict): toe off moments
        heel_strikes (dict): heel strikes moment
        side (string): side to focus on, either 'R' or 'L'.

    Returns:
        corrected data array, in the same shape as given untreated data.

    """
    df_corrected = df.copy()

    if side == 'R':
        to_list = toe_offs['R']
        hs_list = heel_strikes['R']
        cols_to_zero = ['ground_force2_vx', 'ground_force2_vy', 'ground_force2_vz',
                        'ground_force2_px', 'ground_force2_py', 'ground_force2_pz',
                        'ground_torque2_x', 'ground_torque2_y', 'ground_torque2_z']
    elif side == 'L':
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
                if col in df_corrected.columns:
                    df_corrected.loc[toe_idx:heel_idx, col] = 0

    return df_corrected


def write_mot_file(df, column_labels, output_file, header_lines=None):
    """
    This function writes a single MOT file with OpenSim-compatible header + data.
    Args:
        df (dict): data
        column_labels (list): list of column names
        output_file (string): path to output file
        header_lines (list of string): list of header lines

    Returns:
        None.

    """
    with open(output_file, 'w') as file:
        if header_lines:
            file.writelines(header_lines)
        else:
            file.write(f"{os.path.basename(output_file)}\n\n")
        file.write("\t".join(column_labels) + "\n")
        df.to_csv(file, sep='\t', index=False, header=False, lineterminator='\n')
    print(f"Wrote MOT file: {output_file}")


# MAIN
if __name__ == "__main__":
    raw_data_path = local.get_raw_mot_path()
    output_path = local.get_corrected_mot_path()
    os.makedirs(output_path, exist_ok=True)
    mot_data = read_mot_files(raw_data_path)

    for file in mot_data:
        mot = MOT.load(raw_data_path, file['fileName'])
        df = pd.DataFrame(file['rawData'], columns=mot.col_names)
        time = df['time']
        fs = 1 / np.mean(np.diff(time))

        print(f"\nProcessing: {file['fileName']}")
        print(f"Sampling frequency: {fs:.2f} Hz")

        # Filter
        cols = [c for c in df.columns if 'time' != c]
        filtered_df = copy.deepcopy(df)
        filtered_df[cols] = filter_grf(df[cols], fs)

        # Baseline correction
        zeroed_df = baseline_correct_debug(filtered_df, 'ground_force2_vy', ['ground_force2_vx', 'ground_force2_vz'])
        zeroed_df = baseline_correct_debug(zeroed_df, 'ground_force1_vy', ['ground_force1_vx', 'ground_force1_vz'])

        # Detect events
        toe_offs = detect_toe_off_from_df(zeroed_df, fs)
        heel_strikes = detect_heel_strikes_from_df(zeroed_df, fs)
        zeroed_df = zero_swing_phase(zeroed_df, toe_offs, heel_strikes, 'R')
        zeroed_df = zero_swing_phase(zeroed_df, toe_offs, heel_strikes, 'L')

        # Save corrected data as .mot file
        output_name = os.path.splitext(file['fileName'])[0]
        output_file = os.path.join(output_path, f"corrected_{output_name}.mot")

        write_mot_file(zeroed_df, zeroed_df.columns.tolist(), output_file, header_lines=file['headerLines'])

        # Plot GRF with toe-offs and heel-strikes
        plt.figure(figsize=(14, 6))
        right_fy = zeroed_df['ground_force2_vy'].values
        left_fy = zeroed_df['ground_force1_vy'].values

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

        plt.title(f"Vertical GRFs with Toe-Offs and Heel Strikes: {file['fileName']}")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{file['fileName']}_vertical_grfs_with_toeoffs_heelstrikes.png"),
                    bbox_inches='tight')
        plt.show()
