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

def matches(method_name, column_names, gait_side):
    """

    Args:
        method_name (string): name of the method
        column_names:
        gait_side (string): side for the gait cycle

    Returns:

    """
    if method_name == 'moments':
        pattern = r"_r_" if gait_side == 'Right' else r"_l_"
        results = ['pelvis_tilt_moment', 'pelvis_list_moment', 'pelvis_rotation_moment']
        regex = re.compile(pattern)
        for name in column_names:
            if regex.search(name):
                results.append(name)
        return results
    return None


def matches_angles(column_names, gait_side):
    results = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']
    rads_columns = ['pelvis_rotation']
    for name in column_names:
        if len(name) > 2 and name[-2] == '_':
            if gait_side[0].lower() == name[-1]:
                results.append(name)
                rads_columns.append(name)
    return results, rads_columns


def filter_signal(data, cutoff=8, fs=100, order=4):
    """Filters the data"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def compute_angular_velocity(ik_data):
    """Computes the angular velocity from the Ik data"""
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


def compute_joint_power(angular_velocity, id_data, gait_side):
    """Computes the joint power from the angular velocity and id data"""
    matched_ik_cols, matched_id_cols = get_matched_columns(angular_velocity.columns, id_data.columns, gait_side)

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


def get_matched_columns(ik_columns, id_columns, gait_side):
    matched_id_columns = matches("moments", id_columns, gait_side)
    matched_ik_columns, _ = matches_angles(ik_columns, gait_side)
    matched_ik_columns.insert(0, "time")
    matched_id_columns.insert(0, "time")
    return matched_ik_columns, matched_id_columns


def process(trial: Trial, power_output_path: str):
    """Computes the JP data from the trial"""
    # Main Paths
    for side in ["Right", "Left"]:
        output_path = os.path.join(power_output_path, side)
        temp_directory = os.path.join(output_path, "temp")
        os.makedirs(temp_directory, exist_ok=True)

        for cycle in trial.gait_cycles[side]:

            """
            if cycle.ik.filepath is None:
                cycle.ik.save(temp_directory)
            ik_path = cycle.ik.filepath

            if cycle.id.filepath is None:
                cycle.id.save(temp_directory)
            id_path = cycle.id.filepath
            """

            angular_velocity = compute_angular_velocity(cycle.ik.data)
            joint_power = compute_joint_power(angular_velocity, cycle.id.data, side)

            output_filename = f"{trial.name}_JP_{side.lower()}_{cycle.num}.csv"
            output_file_path = os.path.join(output_path, output_filename)
            joint_power.to_csv(output_file_path, index=False)
            cycle.add_joint_power(output_file_path, joint_power)

            # plt.plot(data['time'], data['ankle_angle_r_power'] if side == "Right" else data['ankle_angle_l_power'])
            # plt.show()

            print(f"Successfully processed: {cycle.ik.filename} -> {output_filename}")
