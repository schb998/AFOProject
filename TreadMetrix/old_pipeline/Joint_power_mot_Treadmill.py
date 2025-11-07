import os
import pandas as pd
import numpy as np
import re
import math
from scipy import interpolate
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import TreadMetrix.paths_access as local

"""
This file is used to compute joint power from processed .mot files of Inverse Kinematics & Dynamics.
    Input: segmented .mot files of the IK data, segmented .mot files of the ID data.
    Output: segmented .csv files of the joint power data.
"""


def read_mot_files(data_path_mot):
    """
    Read data from a .mot file.

    Args:
        data_path_mot (string): Path to the .mot file.

    Returns:
        Data from the .mot file, in the form of a list with elements fileName (string), rawData (dataFrame) and colNames (list of the columns' names).

    """
    motion_data_list = []
    file_list = sorted(f for f in os.listdir(data_path_mot) if f.endswith('.mot'))

    for filename in file_list:
        file_path = os.path.join(data_path_mot, filename)
        print(f"Reading MOT file: {file_path}")

        try:
            with open(file_path, 'r') as file:

                for _ in range(6):  # Skip the first 6 header rows
                    next(file)
                data = pd.read_csv(file, sep=r'\s+')
            motion_data_list.append({
                'fileName': filename,
                'rawData': data,
                'colNames': data.columns.to_list()
            })


        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return motion_data_list


# todo check usefulness of moments argument
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


def filter_signal(data, cutoff=8, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def compute_angular_velocity(ik_data, trial_name):
    ik_data_radians = ik_data.copy()
    ik_data_radians.iloc[:, 1:] *= (np.pi / 180)

    x_time = np.linspace(0, 100, len(ik_data_radians))
    angular_velocity = np.zeros_like(ik_data_radians.iloc[:, 1:].values)

    for col_idx, col_name in enumerate(ik_data_radians.columns[1:]):
        spline = interpolate.InterpolatedUnivariateSpline(x_time, ik_data_radians[col_name], k=3)
        angular_velocity[:, col_idx] = spline.derivative()(x_time)

    filtered_angular_velocity = filter_signal(angular_velocity)
    angular_velocity_df = pd.DataFrame(filtered_angular_velocity, columns=ik_data_radians.columns[1:])
    angular_velocity_df.insert(0, "time", x_time)
    return angular_velocity_df


def compute_joint_power(angular_velocity, id_data, gaitcycle):
    matched_ik_cols, matched_id_cols = get_matched_columns(angular_velocity.columns, id_data.columns, gaitcycle)

    # Filtered angular velocity
    angular_velocity_filtered = angular_velocity[matched_ik_cols]

    # Filter joint moments
    id_data_filtered = id_data[matched_id_cols].rename(columns=lambda x: x.replace("_moment", ""))
    id_data_filtered.iloc[:, 1:] = filter_signal(id_data_filtered.iloc[:, 1:].values, cutoff=6, fs=100, order=4)

    # Convert to NumPy arrays
    angular_velocity_values = angular_velocity_filtered.iloc[:, 1:].values
    moment_values = id_data_filtered.iloc[:, 1:].values

    # Trim to match length
    min_len = min(len(angular_velocity_values), len(moment_values))
    angular_velocity_values = angular_velocity_values[:min_len]
    moment_values = moment_values[:min_len]
    time_values = angular_velocity_filtered["time"].values[:min_len]

    # Compute joint power
    joint_power_values = angular_velocity_values * moment_values
    power_column_names = [col + "_power" for col in matched_ik_cols[1:]]
    joint_power_df = pd.DataFrame(joint_power_values, columns=power_column_names)
    joint_power_df.insert(0, "time", time_values)

    return joint_power_df


def get_matched_columns(ik_columns, id_columns, gaitcycle):
    matched_id_columns = matches("moments", id_columns, gaitcycle)
    matched_ik_columns, _ = matches_angles(ik_columns, gaitcycle)
    matched_ik_columns.insert(0, "time")
    matched_id_columns.insert(0, "time")
    return matched_ik_columns, matched_id_columns


# Main Paths
ik_folder = local.get_ik_results_path()
id_folder = local.get_id_results_path()
power_output_folder = local.get_power_filtered_path()

for side in ["Right", "Left"]:
    ik_path = os.path.join(ik_folder, side)
    id_path = os.path.join(id_folder, side)
    output_path = os.path.join(power_output_folder, side)
    os.makedirs(output_path, exist_ok=True)

    ik_mot_files = read_mot_files(ik_path)
    id_mot_files = read_mot_files(id_path)

    for ik_data_dict, id_data_dict in zip(ik_mot_files, id_mot_files):
        ik_filename = ik_data_dict['fileName']
        id_filename = id_data_dict['fileName']
        ik_data = ik_data_dict['rawData']
        id_data = id_data_dict['rawData']

        angular_velocity = compute_angular_velocity(ik_data, ik_filename)
        joint_power = compute_joint_power(angular_velocity, id_data, side)

        output_filename = ik_filename.replace("IK", "Power").replace(".mot", ".csv")
        output_file_path = os.path.join(output_path, output_filename)
        joint_power.to_csv(output_file_path, index=False)

        print(f"Successfully processed: {ik_filename} -> {output_filename}")
