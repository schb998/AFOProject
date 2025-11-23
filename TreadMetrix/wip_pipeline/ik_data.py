import os
import pathlib
import pandas as pd
import resources.paths.paths_access as local
import numpy as np
from scipy.signal import butter, filtfilt
from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
import opensim as osim

"""
This file is used to compute Inverse Kinematic data.
    Inputs: segmented .trc file, corresponding .osim file, array of the markers used.
    Output: segmented .mot files of the IK data.
"""


def filter_signals(data, fs=100, cutoff=6, order=2):
    """
    Filter the signal according to the Butterworth method.

    Args:
        data (array): signal to be filtered
        fs (int): sampling frequency
        cutoff (int): half cycles
        order (int): order of the filter

    Returns:
        array: filtered signal

    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def read_mot_storage(filepath: str) -> (tuple, str):
    """
    Read a .mot file from the storage path.

    Args:
        filepath: string, path to the mot file.

    Returns:
        String array: labels of the .mot file
        Array: data of the .mot file

    """
    storage = osim.Storage(filepath)
    label_array = storage.getColumnLabels()
    labels = [label_array.get(label) for label in range(label_array.getSize())]

    time_vec = []
    data_vec = []
    for v in range(storage.getSize()):
        row = storage.getStateVector(v)
        time_vec.append(row.getTime())
        data_array = row.getData()
        data_row = [data_array.get(j) for j in range(data_array.getSize())]
        data_vec.append(data_row)

    data = np.array(data_vec)
    time_vec = np.array(time_vec).reshape(-1, 1)
    return labels, time_vec, data


def set_up_ik_tool(model_file, marker_data, start_time, end_time):
    """
    Set up OpenSim's Inverse Kinematics tool.

    Args:
        model_file:
        marker_data:
        start_time:
        end_time:

    Returns:

    """
    tool = osim.InverseKinematicsTool()
    tool.set_model_file(model_file)
    tool.setMarkerDataFileName(marker_data)
    tool.setStartTime(start_time)
    tool.setEndTime(end_time)
    return tool


def marker_tasks(tool, markers, do_not_include_list):
    """
    Setup OpenSim's taskset with gicen markers.

    Args:
        tool:
        markers:
        do_not_include_list:

    Returns:

    """
    taskset = tool.getIKTaskSet()
    for m in markers:
        task = osim.IKMarkerTask()
        task.setName(m)
        task.setApply(m not in do_not_include_list)
        task.setWeight(1)
        taskset.cloneAndAppend(task)
    return taskset



def process(segmented_trcs: dict[str, list[TRC]], filename: str):
    model_file = local.get_scaled_model_file()
    ik_results_path = os.path.join(local.get_ik_results_path(), filename)
    os.makedirs(ik_results_path, exist_ok=True)

    marker_names = [
        'Sternum', 'LShoulder', 'RShoulder', 'LASIS', 'RASIS', 'RPSIS', 'LPSIS',
        'RFibula', 'RShank', 'RAnkleLateral', 'RToe', 'LToe', 'RMT5', 'RMT2', 'RHeel',
        'LFibula', 'LShank', 'LAnkleLateral', 'LMT5', 'LMT2', 'LHeel', 'RKneeLateral',
        'LAnkleMedial', 'LKneeLateral', 'RAnkleMedial', 'LKneeMedial', 'RKneeMedial'
    ]
    do_not_include = ['RKneeMedial', 'RAnkleMedial', 'RToe', 'LKneeMedial', 'LAnkleMedial', 'LToe']

    for side in ["Right", "Left"]:
        trcs = segmented_trcs[side]
        ik_output_path = os.path.join(ik_results_path, side)
        temp_trc_directory = os.path.join(ik_output_path, "temp")
        os.makedirs(temp_trc_directory, exist_ok=True)

        cycle_num = 1
        for trc in trcs:
            print(f"Processing {side}/{trc.filename}...")

            trc.save(temp_trc_directory)
            trc_full_path = os.path.join(temp_trc_directory, trc.filename)

            # Setup IK Tool
            ik_tool = set_up_ik_tool(model_file, trc_full_path, float(trc.data['Time'].iloc[0]),
                                     float(trc.data['Time'].iloc[-1]))

            # Name format
            cycle_name = f"{side.lower()}_cycle_{cycle_num}"
            mot_name = f"{cycle_name}.mot"
            mot_path = os.path.join(ik_output_path, mot_name)
            ik_tool.setOutputMotionFileName(mot_path)

            # Add marker tasks
            task_set = marker_tasks(ik_tool, marker_names, do_not_include)

            ik_tool.run()

            # Read and filter:
            if os.path.exists(mot_path):
                header, time_vec, data = read_mot_storage(mot_path)
                data = filter_signals(data)
                data = np.hstack((time_vec, data))

                mot = MOT.load_from_mot(mot_path, separator=r"\t")
                mot.data = pd.DataFrame(data)
                mot.save(ik_output_path)

            else:
                print(f"IK failed for: {trc.filename}")

            os.remove(trc_full_path)
            cycle_num = cycle_num + 1

        pathlib.Path.rmdir(pathlib.Path(temp_trc_directory))

    print("\nAll IK trials processed and saved as filtered .mot files.")
