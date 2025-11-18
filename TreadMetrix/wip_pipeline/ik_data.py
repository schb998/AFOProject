import os
import pandas as pd
import resources.paths.paths_access as local
from TreadMetrix.wip_pipeline.osim_gestion import configure_opensim
import numpy as np
from scipy.signal import butter, filtfilt
from resources.file_types.trc import TRC
from ptb.util.osim.osim_store import OSIMStorage, HeadersLabels
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
    return labels, np.hstack((time_vec, data))


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
        taskset.append(task)
    return taskset


if __name__ == "__main__":
    # Setup OpenSim
    configure_opensim()

    # Paths
    model_file = local.get_scaled_model_file()
    trc_path = local.get_segmented_trc_path()
    ik_results_path = local.get_ik_results_path()

    # Marker setup
    marker_names = [
        'Sternum', 'LShoulder', 'RShoulder', 'LASIS', 'RASIS', 'RPSIS', 'LPSIS',
        'RFibula', 'RShank', 'RAnkleLateral', 'RToe', 'LToe', 'RMT5', 'RMT2', 'RHeel',
        'LFibula', 'LShank', 'LAnkleLateral', 'LMT5', 'LMT2', 'LHeel', 'RKneeLateral',
        'LAnkleMedial', 'LKneeLateral', 'RAnkleMedial', 'LKneeMedial', 'RKneeMedial'
    ]
    do_not_include = ['RKneeMedial', 'RAnkleMedial', 'RToe', 'LKneeMedial', 'LAnkleMedial', 'LToe']

    for side in ["Right", "Left"]:
        trc_side_path = os.path.join(trc_path, side)
        out_side_path = os.path.join(ik_results_path, side)
        trc_files = sorted([f for f in os.listdir(trc_side_path) if f.endswith(".trc")])

        for i, trc_file in enumerate(trc_files, 1):
            print(f"Processing {side}/{trc_file}...")

            trc_full_path = os.path.join(trc_side_path, trc_file)
            trc = TRC.load_from_trc(trc_full_path)

            # Setup IK Tool
            ik_tool = set_up_ik_tool(model_file, trc_full_path, float(trc.data['Time'].iloc[0]),
                                     float(trc.data['Time'].iloc[-1]))

            # Name format
            cycle_name = f"{side.lower()}_cycle_{i}"
            mot_path_temp = os.path.join(out_side_path, f"{cycle_name}_temp.mot")
            mot_path_final = os.path.join(out_side_path, f"{cycle_name}.mot")
            ik_tool.setOutputMotionFileName(mot_path_temp)

            # Add marker tasks
            task_set = marker_tasks(ik_tool, marker_names, do_not_include)

            ik_tool.run()

            if not os.path.exists(mot_path_temp):
                print(f"IK failed for {trc_file}")
                continue

            # Read and filter
            headers, raw_data = read_mot_storage(mot_path_temp)
            time = raw_data[:, 0]
            signals = raw_data[:, 1:]
            filtered = filter_signals(signals)
            filtered_data = np.column_stack((time, filtered))
            h = OSIMStorage.simple_header_template()
            filename = os.path.split(mot_path_temp)[1]

            h[HeadersLabels.trial] = filename[:filename.rindex('.')]
            h[HeadersLabels.version] = 1
            h[HeadersLabels.nRows] = filtered_data.shape[0]
            h[HeadersLabels.nColumns] = filtered_data.shape[1]
            h[HeadersLabels.inDegrees] = True
            mot = OSIMStorage.create(data=pd.DataFrame(data=filtered_data, columns=headers), header=h,
                                     filename=filename)
            mot.write(mot_path_final)

            os.remove(mot_path_temp)

    print("\nAll IK trials processed and saved as filtered .mot files.")
