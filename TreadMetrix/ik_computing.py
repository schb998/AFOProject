import array
import os
import pathlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from resources.file_types.mot import MOT
import opensim as osim
from resources.trial_class import Trial

"""
This file is used to compute Inverse Kinematic data.
"""

# todo: set the _ik_marker_errors.sto output of the ik tool in the given ik folder


def filter_signals(data: np.ndarray, fs: float = 100.0, cutoff: float = 6.0, order: int = 4) -> np.ndarray:
    """
    Filter the signal according to the Butterworth method (zero-phase, forward-backward).

    Recommended parameters based on spectrum analysis of P03 gait data:
      - Kinematics (IK angles): 4th-order Butterworth, 6 Hz cutoff, zero-phase

    Args:
        data:   array, signal to be filtered
        fs:     float, sampling frequency (Hz). Must be computed from the actual data.
        cutoff: float, cutoff frequency (Hz). Default 6.0 Hz per spectrum analysis.
        order:  int,   filter order. Default 4 (effective 8th-order with filtfilt).

    Returns:
        array: filtered signal
    """
    if len(data) <= 3:
        return data
    nyq = 0.5 * fs
    normal_cutoff = min(0.99, cutoff / nyq)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    padlen = min(15, len(data) - 1)
    return filtfilt(b, a, data, axis=0, padlen=padlen)


def read_mot_storage(filepath: str) -> (list[str], np.array, np.array):
    """
    Read a MOT file from the storage path.

    Args:
        filepath: string, path to the MOT file.

    Returns:
        String list: labels of the MOT file
        np.array: time vector of the MOT data
        np.array: MOT data

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


def set_up_ik_tool(model_file, marker_data, start_time, end_time) -> osim.InverseKinematicsTool:
    """
    Set up OpenSim's Inverse Kinematics tool.

    Args:
        model_file: str, path to the scaled OpenSim Model
        marker_data: str, path to the OpenSim marker file (TRC).
        start_time: float, time at the first frame to process
        end_time: float, time at the last frame to process

    Returns:
        set-up OpenSim's InverseKinematicsTool

    """
    tool = osim.InverseKinematicsTool()
    tool.set_model_file(model_file)
    tool.setMarkerDataFileName(marker_data)
    tool.setStartTime(start_time)
    tool.setEndTime(end_time)
    return tool


def marker_tasks(tool: osim.InverseKinematicsTool, markers: list[str], do_not_include_list: list[str]) \
        -> osim.IKMarkerTask:
    """
    Setup OpenSim's taskset with given markers.

    Args:
        tool: OpenSim Ik tool.
        markers: string list, list of makers to add to the task
        do_not_include_list: string list, list of markers to put aside

    Returns:
        OpenSim Marker Task Set
    """
    taskset = tool.getIKTaskSet()
    for m in markers:
        task = osim.IKMarkerTask()
        task.setName(m)
        task.setApply(m not in do_not_include_list)
        task.setWeight(1)
        taskset.cloneAndAppend(task)
    return taskset


def process(trial: Trial, scaled_model_file_path: str, ik_result_path: str, save: bool = True):
    """Pipeline to compute the Internal Kinematics results from a trial's gait cycles.

    Args:
        trial: Trial object, trial to process
        scaled_model_file_path: str, path to the scaled model file
        ik_result_path: str, where to save the resulting IK files
        save: bool, whether to keep the saved IK files or not

    Returns:
        None
    """

    os.makedirs(ik_result_path, exist_ok=True)

    marker_names = [
        'CLAV', 'RACR1', 'LACR1', 'RASI', 'LASI', 'RPSI', 'LPSI',
        'RTH1', 'RTH2', 'RTH3', 'RLFC', 'RMFC',
        'RTB1', 'RTB2', 'RTB3', 'RLMAL', 'RMMAL',
        'RCAL', 'RMT1', 'RMT5', 'RToe',
        'LTH1', 'LTH2', 'LTH3', 'LLFC', 'LMFC',
        'LTB1', 'LTB2', 'LTB3', 'LLMAL', 'LMMAL',
        'LCAL', 'LMT1', 'LMT5', 'LToe',
        'T10', 'C7', 'RHJC', 'LHJC'
    ]
    do_not_include = ['RMFC', 'RMMAL', 'RToe', 'LMFC', 'LMMAL', 'LToe']

    for side in ["Right", "Left"]:
        ik_output_path = os.path.join(ik_result_path, side)
        temp_directory = os.path.join(ik_output_path, "temp")
        os.makedirs(temp_directory, exist_ok=True)

        for cycle in trial.gait_cycles[side]:
            trc = cycle.trc

            print(f"Processing {side}/{trc.filename}...")

            if cycle.trc.filepath is None or not os.path.exists(cycle.trc.filepath):
                trc.save(temp_directory)
            trc_full_path = cycle.trc.filepath

            # Setup IK Tool
            ik_tool = set_up_ik_tool(scaled_model_file_path, trc_full_path, float(trc.data['Time'].iloc[0]),
                                     float(trc.data['Time'].iloc[-1]))

            # Name format
            cycle_name = f"{trial.name}_{side}_cycle{cycle.num}"
            mot_name = f"{cycle_name}.mot"
            mot_path = os.path.join(ik_output_path, mot_name)
            ik_tool.setOutputMotionFileName(mot_path)

            # Add marker tasks
            task_set = marker_tasks(ik_tool, marker_names, do_not_include)

            ik_tool.run()

            # Read and filter:
            if os.path.exists(mot_path):
                header, time_vec, data = read_mot_storage(mot_path)
                # Compute fs dynamically from the actual time vector
                fs_ik = 1.0 / float(np.mean(np.diff(time_vec.ravel())))
                # Kinematic filtering at 6.0 Hz cut-off frequency
                data = filter_signals(data, fs=fs_ik, cutoff=6.0, order=4)
                data = np.hstack((time_vec, data))

                mot = MOT.load_from_mot(mot_path)
                data = pd.DataFrame(data)
                data.columns = header
                mot.update_data(data)

                mot.save(ik_output_path)
                cycle.add_ik(inverse_kinematic_path=mot_path, ik_object=mot)

                # plt.plot(cycle.ik.data['time'], cycle.ik.data['ankle_angle_r'])
                # plt.show()

            else:
                print(f"IK failed for: {trc.filename}")

            for file in os.listdir(temp_directory):
                os.remove(os.path.join(temp_directory, file))

        try:
            pathlib.Path.rmdir(pathlib.Path(temp_directory))
        except OSError:
            print(f"Error deleting temporary directory {temp_directory}.")

    print("\nAll IK trials processed.")
