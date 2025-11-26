from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
import resources.paths.paths_access as local
import os
from copy import deepcopy
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import numpy as np

# todo: check segment_at_heel_strikes function when mot_frame_rate is None // trc_rate is not in trc.metadata


def filter_grf(mot: MOT, fs: float) -> None:
    """Filters data of a MOT object with a Butterworth filter.

    Args:
        mot: MOT object whose data is to be filtered.
        fs: sampling frequency.
    """
    b, a = butter(6, (12 / (fs / 2)), btype='low', output='ba')
    filtered_df = deepcopy(mot.data)
    for col in mot.data.columns.tolist():
        filtered_df[col] = filtfilt(b, a, mot.data[col])
    mot.data = filtered_df


def baseline_correct_debug(mot_object: MOT, fz_col: str, related_cols: list[str], output_path: str = None,
                           show: bool = False) \
        -> None:
    """Corrects the baseline of one of the columns of the mot data.

    Saves plot of the baseline correction.

    Args:
        mot_object: data to process.
        fz_col: name of the column to correct.
        related_cols: other columns to consider.
        output_path: output path for plot save. Optional. If None, plot is not saved.
        show: whether to show the figure when method is called.
    """
    fy = mot_object.data[fz_col]
    corrected_df = deepcopy(mot_object.data)
    valley_indices, _ = find_peaks(-fy)
    swing_valleys = valley_indices[fy[valley_indices] < 0]

    print(f"\nCorrecting {fz_col}")
    print(f"Number of swing valleys below 0N: {len(swing_valleys)}")

    if len(swing_valleys) == 0:
        print("No valleys found below zero. Skipping correction.")

    baseline = abs(np.median(fy[swing_valleys]))
    print(f"Baseline offset to add: {baseline:.2f}")
    corrected_df[fz_col] = fy + baseline

    for col in related_cols:
        related = mot_object.data[col]
        offset = np.median(related[swing_valleys])
        corrected_df[col] = related - offset if offset > 0 else related + abs(offset)
        print(f"Offset for {col}: {offset:.2f}")

    mot_object.data = corrected_df

    if (output_path is not None) or show:
        time_scale = mot_object.data['time'] if 'time' in mot_object.data.columns.tolist() else np.arange(len(fy))
        plt.figure(figsize=(12, 4))
        plt.plot(time_scale, fy, label='Original', alpha=0.7)
        plt.scatter(time_scale[swing_valleys], fy[swing_valleys], color='red', label='Swing Valleys')
        plt.plot(time_scale, corrected_df[fz_col], label='Corrected', alpha=0.8)
        plt.title(f"{fz_col} Baseline Correction")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()
        plt.grid(True)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            file_name = f"{mot_object.filename.replace('.mot', '')}_baseline_correction_{fz_col}.png"
            plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight')

        if show:
            plt.show()


def detect_toe_offs(zeroed_mot: MOT, fs: float, threshold: float = 20) -> dict[str, list[int]]:
    """Detects the gait cycles' "toe off" points.

    Args:
        zeroed_mot: data to process.
        fs: sampling frequency.
        threshold: threshold to use for detection.

    Returns:
        Dictionary of the data's toe offs, listed by side.

    """
    toe_offs = {'R': [], 'L': []}
    prominence = 15
    distance = int(fs / 10)
    height = 200

    if 'ground_force2_vy' in zeroed_mot.data.columns.tolist():
        rzf = zeroed_mot.data['ground_force2_vy']
        r_indexes, _ = find_peaks(rzf, prominence=prominence, distance=distance, height=height)
        peak = 0
        while peak < np.shape(r_indexes)[0]:
            idx_start = r_indexes[peak]
            below_thresh = np.where(rzf[idx_start:] < threshold)[0]
            if len(below_thresh) == 0:
                break
            toe_offs['R'].append(idx_start + below_thresh[0])
            next_peaks = np.where(r_indexes > toe_offs['R'][-1])[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]

    if 'ground_force1_vy' in zeroed_mot.data.columns.tolist():
        lzf = zeroed_mot.data['ground_force1_vy']
        l_indexes, _ = find_peaks(lzf, prominence=prominence, distance=distance, height=height)
        peak = 0
        while peak < np.shape(l_indexes)[0]:
            idx_start = l_indexes[peak]
            below_thresh = np.where(lzf[idx_start:] < threshold)[0]
            if len(below_thresh) == 0:
                break
            toe_offs['L'].append(idx_start + below_thresh[0])
            next_peaks = np.where(l_indexes > toe_offs['L'][-1])[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]

    return toe_offs


def detect_heel_strikes(zeroed_mot: MOT, fs: float, threshold: float = 20) -> dict[str, list[int]]:
    """Detects the gait cycles' "heel strikes" points.
    Args:
        zeroed_mot: data to process.
        fs: sampling frequency.
        threshold: threshold to use for detection.

    Returns:
        Dictionary of the data's heel strikes, listed by side.

    """
    heel_contacts = {'R': [], 'L': []}
    distance = int(fs / 2)
    prominence = 14
    height = -100

    if 'ground_force2_vy' in zeroed_mot.data.columns.tolist():
        rzf = zeroed_mot.data['ground_force2_vy']
        r_indexes, _ = find_peaks(-rzf, prominence=prominence, distance=distance, height=height)
        rest = len(rzf)
        peak = 0
        while peak < np.shape(r_indexes)[0] and rest > distance:
            idx_start = r_indexes[peak]
            above_thresh = np.where(rzf[idx_start:] > threshold)[0]
            if len(above_thresh) == 0:
                break
            heel_idx = idx_start + above_thresh[0]
            heel_contacts['R'].append(heel_idx)
            next_peaks = np.where(r_indexes > heel_idx)[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]
            rest = len(rzf[r_indexes[peak]:])

    if 'ground_force1_vy' in zeroed_mot.data.columns.tolist():
        lzf = zeroed_mot.data['ground_force1_vy']
        l_indexes, _ = find_peaks(-lzf, prominence=prominence, distance=distance, height=height)
        rest = len(lzf)
        peak = 0
        while peak < np.shape(l_indexes)[0] and rest > distance:
            idx_start = l_indexes[peak]
            above_thresh = np.where(lzf[idx_start:] > threshold)[0]
            if len(above_thresh) == 0:
                break
            heel_idx = idx_start + above_thresh[0]
            heel_contacts['L'].append(heel_idx)
            next_peaks = np.where(l_indexes > heel_idx)[0]
            if len(next_peaks) == 0:
                break
            peak = next_peaks[0]
            rest = len(lzf[l_indexes[peak]:])

    return heel_contacts


def zero_swing_phase(mot_df: MOT, toe_offs: dict[str, list[int]], heel_strikes: dict[str, list[int]], side: str)\
        -> None:
    """Sets GRF and related columns to zero between toe-off and next heel strike.

    Args:
        mot_df: data to process.
        toe_offs: toe off moments, listed in directory by side.
        heel_strikes: heel strikes moment, listed in directory by side.
        side: side to focus on. Method raises ValueError if invalid.

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


def plot_grf_details(mot: MOT, heel_strikes: dict[str, list[int]], toe_offs: dict[str, list[int]],
                     output: str, show: bool = False) -> None:
    """Saves plot of the vertical forces with toe offs and heel strikes.

    Args:
        mot: MOT object of the data.
        heel_strikes: heel strikes moment, listed in directory by side.
        toe_offs: toe offs moment, listed in directory by side
        output: output directory name.
        show: whether to show the figure when method is called.

    """
    plt.figure(figsize=(14, 6))
    time_scale = mot.data['time'] if 'time' in mot.data.columns.tolist() else np.arange(mot.data.shape[0])
    right_fy = mot.data['ground_force2_vy']
    left_fy = mot.data['ground_force1_vy']

    plt.plot(time_scale, right_fy, label='Right Fy', alpha=0.7)
    plt.plot(time_scale, left_fy, label='Left Fy', alpha=0.7)

    # Toe-offs
    plt.scatter([time_scale[i] for i in toe_offs['R']], [right_fy[i] for i in toe_offs['R']],
                color='red', marker='x', label='Right Toe-Offs')
    plt.scatter([time_scale[i] for i in toe_offs['L']], [left_fy[i] for i in toe_offs['L']],
                color='green', marker='x', label='Left Toe-Offs')

    # Heel strikes
    plt.scatter([time_scale[i] for i in heel_strikes['R']], [right_fy[i] for i in heel_strikes['R']],
                color='blue', marker='o', label='Right Heel Strikes')
    plt.scatter([time_scale[i] for i in heel_strikes['L']], [left_fy[i] for i in heel_strikes['L']],
                color='purple', marker='o', label='Left Heel Strikes')

    plt.title(f"Vertical GRFs with Toe-Offs and Heel Strikes: {mot.filename}")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output, exist_ok=True)
    plt.savefig(os.path.join(output,
                             f"{mot.filename.replace('.mot', '')}_vertical_grfs_with_toeoffs_heelstrikes.png"),
                bbox_inches='tight')
    if show:
        plt.show()


def segment_at_heel_strikes(mot: MOT, heel_strike_moments: dict[str, list[int]], mot_frame_rate: float = None,
                            trc: TRC = None, save: str = None) -> dict[str, dict[str, list[MOT | TRC]]]:
    """Segment MOT (and matching TRC) object(s) according to heel_strikes.

    Args:
        mot: MOT object to process.
        heel_strike_moments: dictionary of heel strikes, listed by side ("R"/"L").
        trc: TRC object matching the given MOT. Optional.
        mot_frame_rate: frame rate of mot_file. Optional.
            Used when trc is not None, for faster results.
        save: Where to save the segmented files. Optional. Don't save if None.

    Returns:
        dict: dictionary of the segmented objects, organized in lists by type (trc/mot) and side.

    """
    # segment mot object:
    right_mots = mot.segment(heel_strike_moments['R'])[1:-1]
    left_mots = mot.segment(heel_strike_moments['L'])[1:-1]
    res = {'mot': {"Right": right_mots, "Left": left_mots}}

    if save is not None:
        path = os.path.join(save, mot.filename.replace('.mot', ''))
        MOT.save_multiple(right_mots, os.path.join(path, "Right"))
        MOT.save_multiple(left_mots, os.path.join(path, "Left"))

    # process trc object:
    if trc is not None:

        # get the list of heel strikes:
        trc_rate = int(trc.metadata['CameraRate']) if 'CameraRate' in trc.metadata.keys() else None
        if mot_frame_rate is None:
            mot_frame_rate = 1 / np.mean(np.diff(mot.data['time']))
        if trc_rate is None:
            trc_rate = 1 / np.mean(np.diff(trc.data['Time']))
        rate_conversion = trc_rate / mot_frame_rate
        trc_heel_strike_moments = {}
        for side in heel_strike_moments:
            trc_heel_strike_moments[side] = []
            for i in range(len(heel_strike_moments[side])):
                trc_heel_strike_moments[side].append(int(heel_strike_moments[side][i] * rate_conversion))

        # segment trc:
        right_trcs = trc.segment(trc_heel_strike_moments['R'])[1:-1]
        left_trcs = trc.segment(trc_heel_strike_moments['L'])[1:-1]
        res['trc'] = {"Right": right_trcs,  "Left": left_trcs}

        if save is not None:
            path = os.path.join(save, mot.filename.replace('.mot', ''))
            TRC.save_multiple(right_trcs, os.path.join(path, "Right"))
            TRC.save_multiple(left_trcs, os.path.join(path, "Left"))

    return res
