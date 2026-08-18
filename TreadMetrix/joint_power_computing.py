import os
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot, pyplot as plt
from scipy import interpolate
from scipy.signal import butter, filtfilt
from resources.trial_class import Trial

"""
This file is used to compute joint power from processed .mot files of Inverse Kinematics & Dynamics.
    Input: segmented .mot files of the IK data, segmented .mot files of the ID data.
    Output: segmented .csv files of the joint power data.
"""


def read_mot_files(file_path):
    """
    Read data from a .mot file.

    Args:
        data_path_mot (string): Path to the .mot file.

    Returns:
        Data from the .mot file, in the form of a list with elements fileName (string), rawData (dataFrame) and colNames (list of the columns' names).

    """
    filename = os.path.basename(file_path)
    try:
        with open(file_path, 'r') as file:

            for _ in range(6):  # Skip the first 6 header rows
                next(file)
            data = pd.read_csv(file, sep=r'\s+')
        return {
            'fileName': filename,
            'rawData': data,
            'colNames': data.columns.to_list()
        }

    except Exception as e:
        print(f"Error reading {filename}: {e}")


def matches(method_name, column_names, gaitcycle):
    """

    Args:
        method_name (string): name of the method
        column_names:
        gaitcycle (string): side for the gait cycle

    Returns:

    """
    if method_name == 'moments':
        pattern = r"_r_" if gaitcycle == 'Right' else r"_l_"
        results = ['pelvis_tilt_moment', 'pelvis_list_moment', 'pelvis_rotation_moment']
        regex = re.compile(pattern)
        for name in column_names:
            if regex.search(name):
                results.append(name)
        return results
    return None


def matches_angles(column_names, gaitcycle):
    results = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']
    rads_columns = ['pelvis_rotation']
    for name in column_names:
        if len(name) > 2 and name[-2] == '_':
            if gaitcycle[0].lower() == name[-1]:
                results.append(name)
                rads_columns.append(name)
    return results, rads_columns


def filter_signal(data, cutoff=6.0, fs=100.0, order=4):
    """
    Low-pass Butterworth filter (zero-phase, forward-backward filtfilt).

    Recommended parameters based on spectrum analysis of P03 gait data:
      - Kinematics (IK angles, angular velocity, joint moments): 4th-order, 4 Hz, zero-phase

    Args:
        data:   array, signal to filter
        cutoff: float, cutoff frequency in Hz (default 4 Hz per spectrum analysis)
        fs:     float, sampling frequency in Hz — compute from data, do NOT hardcode
        order:  int,   Butterworth filter order (default 4; effective 8th-order with filtfilt)
    """
    if len(data) <= 3:
        return data
    nyq = 0.5 * fs
    normal_cutoff = min(0.99, cutoff / nyq)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    padlen = min(15, len(data) - 1)
    return filtfilt(b, a, data, axis=0, padlen=padlen)


def compute_angular_velocity(ik_data, trial_name):
    ik_data_radians = ik_data.copy()
    ik_data_radians.iloc[:, 1:] *= (np.pi / 180)

    x_time = ik_data_radians['time'].values
    angular_velocity = np.zeros_like(ik_data_radians.iloc[:, 1:].values)

    for col_idx, col_name in enumerate(ik_data_radians.columns[1:]):
        spline = interpolate.InterpolatedUnivariateSpline(x_time, ik_data_radians[col_name], k=3)
        angular_velocity[:, col_idx] = spline.derivative()(x_time)

    # Compute fs dynamically from the IK time vector
    fs_ik = 1.0 / float(np.mean(np.diff(x_time)))
    # Single-pass spline derivative (bypass secondary 6.0 Hz filter to prevent double-filtering)
    # filtered_angular_velocity = filter_signal(angular_velocity, cutoff=6.0, fs=fs_ik, order=4)
    angular_velocity_df = pd.DataFrame(angular_velocity, columns=ik_data_radians.columns[1:])
    angular_velocity_df.insert(0, "time", x_time)
    return angular_velocity_df


def compute_joint_power(angular_velocity, id_data, gaitcycle):
    matched_ik_cols, matched_id_cols = get_matched_columns(angular_velocity.columns, id_data.columns, gaitcycle)

    # Filtered angular velocity (includes 'time' at position 0)
    angular_velocity_filtered = angular_velocity[matched_ik_cols]

    # Filter joint moments and strip '_moment' suffix so column names equal the IK joint names.
    # After renaming, id_data_filtered has the same joint-name columns as angular_velocity_filtered.
    id_data_filtered = id_data[matched_id_cols].rename(columns=lambda x: x.replace("_moment", ""))
    t_id_full = id_data_filtered['time'].values.astype(float)
    fs_id = 1.0 / float(np.mean(np.diff(t_id_full))) if len(t_id_full) > 1 else 100.0
    # Spectrum analysis recommendation: 4th-order Butterworth, 6 Hz cutoff, zero-phase
    id_data_filtered.iloc[:, 1:] = filter_signal(id_data_filtered.iloc[:, 1:].values, cutoff=6.0, fs=fs_id, order=4)

    # -----------------------------------------------------------------------
    # Timestamp Alignment using Exact Linear Interpolation
    # -----------------------------------------------------------------------
    t_ik = angular_velocity_filtered['time'].values.astype(float)
    t_id = id_data_filtered['time'].values.astype(float)

    # If one file has absolute trial time (e.g. 15.2s) while the other was reset to 0.0s,
    # align them by shifting t_id to match t_ik's time origin
    if abs(t_ik[0] - t_id[0]) > 0.5:
        dur_ik = t_ik[-1] - t_ik[0]
        dur_id = t_id[-1] - t_id[0]
        if abs(dur_ik - dur_id) < 0.2:
            t_id = t_id - t_id[0] + t_ik[0]

    # Compute overlapping time window
    t_start = max(t_ik[0], t_id[0])
    t_end   = min(t_ik[-1], t_id[-1])

    if t_end <= t_start:
        t_target = t_ik
        ik_mask = np.ones(len(t_ik), dtype=bool)
    else:
        ik_mask = (t_ik >= t_start) & (t_ik <= t_end)
        t_target = t_ik[ik_mask]

    n_samples = len(t_target)
    if n_samples < 2:
        t_target = t_ik
        ik_mask = np.ones(len(t_ik), dtype=bool)
        n_samples = len(t_target)

    # -----------------------------------------------------------------------
    # Compute power JOINT-BY-JOINT using NAMED column lookup & timestamp interpolation.
    # OpenSim IK and ID .mot files can return joint columns in different orders.
    # Looking up each joint by name guarantees the correct pairing regardless
    # of what order either file happens to list its columns.
    # -----------------------------------------------------------------------
    joint_names      = [col for col in matched_ik_cols if col != "time"]
    power_column_names = [j + "_power" for j in joint_names]

    import resources.paths.paths_access as c
    subject_weight = c.get_subject_weight()

    power_data = {}
    for joint in joint_names:
        if joint not in angular_velocity_filtered.columns:
            print(f"  [WARNING] Joint '{joint}' missing from angular-velocity DataFrame — skipping.")
            power_data[joint + "_power"] = np.zeros(n_samples)
            continue
        if joint not in id_data_filtered.columns:
            print(f"  [WARNING] Joint '{joint}' missing from moment DataFrame — skipping.")
            power_data[joint + "_power"] = np.zeros(n_samples)
            continue

        av_vals = angular_velocity_filtered.loc[ik_mask, joint].values.astype(float)
        id_vals_raw = id_data_filtered[joint].values.astype(float)

        # Interpolate moments onto exact IK timestamps
        moment_interp = np.interp(t_target, t_id, id_vals_raw)

        p = av_vals * moment_interp
        if subject_weight is not None and subject_weight > 0:
            if np.max(np.abs(moment_interp)) > 10.0:
                p = p / subject_weight
        power_data[joint + "_power"] = p

    # Time-normalize to exactly 101 points (0 % to 100 % of gait cycle)
    normalized_power = np.zeros((101, len(power_column_names)))
    percentage      = np.linspace(0, 100, 101)
    orig_percentage = np.linspace(0, 100, n_samples)

    for i, col in enumerate(power_column_names):
        spline = interpolate.InterpolatedUnivariateSpline(orig_percentage, power_data[col], k=3)
        normalized_power[:, i] = spline(percentage)

    joint_power_df = pd.DataFrame(normalized_power, columns=power_column_names)
    joint_power_df.insert(0, "time_percent", percentage)

    return joint_power_df


def get_matched_columns(ik_columns, id_columns, gaitcycle):
    matched_id_columns = matches("moments", id_columns, gaitcycle)
    matched_ik_columns, _ = matches_angles(ik_columns, gaitcycle)
    matched_ik_columns.insert(0, "time")
    matched_id_columns.insert(0, "time")
    return matched_ik_columns, matched_id_columns


def process(trial: Trial, power_output_path: str):
    # Main Paths
    for side in ["Right", "Left"]:
        output_path = os.path.join(power_output_path, side)
        temp_directory = os.path.join(output_path, "temp")
        os.makedirs(temp_directory, exist_ok=True)

        for cycle in trial.gait_cycles[side]:

            if cycle.ik is None or cycle.id is None:
                print(f"Skipping Joint Power for {side} cycle {cycle.num}: missing IK or ID data.")
                continue

            angular_velocity = compute_angular_velocity(cycle.ik.data, trial.name)
            joint_power = compute_joint_power(angular_velocity, cycle.id.data, side)

            output_filename = f"{trial.name}_{side}_cycle{cycle.num}.csv"
            output_file_path = os.path.join(output_path, output_filename)
            joint_power.to_csv(output_file_path, index=False)
            cycle.add_joint_power(output_file_path, joint_power)

            # Plot is handled at the end
            print(f"Successfully processed: {cycle.ik.filename} -> {output_filename}")

        # Plot average ankle power for this side (skipped per user request)
        pass
