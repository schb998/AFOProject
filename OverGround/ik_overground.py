import os
import pathlib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import opensim as osim
import sys
# Add project root to path to allow importing 'resources'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC

""" change the markerset to the one you used, the script filters the marker trajectories first, reads the segmented trc files and opensim ik tool """
""" first compute the ik and then filter"""""

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def filter_signals(data: np.ndarray, fs: int = 100, cutoff: int = 6, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def read_mot_storage(filepath: str) -> (list[str], np.ndarray, np.ndarray):
    storage = osim.Storage(filepath)
    label_array = storage.getColumnLabels()
    labels = [label_array.get(i) for i in range(label_array.getSize())]

    time_vec, data_vec = [], []
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
    tool = osim.InverseKinematicsTool()
    tool.set_model_file(model_file)
    tool.setMarkerDataFileName(marker_data)
    tool.setStartTime(start_time)
    tool.setEndTime(end_time)
    return tool


def marker_tasks(tool: osim.InverseKinematicsTool, markers: list[str], do_not_include_list: list[str]):
    taskset = tool.getIKTaskSet()
    for m in markers:
        task = osim.IKMarkerTask()
        task.setName(m)
        task.setApply(m not in do_not_include_list)
        task.setWeight(1)
        taskset.cloneAndAppend(task)
    return taskset


def iter_segmented_trcs(segmented_root: str):
    """
    Expects structure:
      segmented/<Trial>/Right/FP2/*.trc
      segmented/<Trial>/Left/FP1/*.trc
    Yields (trial, side, fp_folder, trc_path)
    """
    for trial_name in os.listdir(segmented_root):
        tdir = os.path.join(segmented_root, trial_name)
        if not os.path.isdir(tdir):
            continue
        for side in ["Right", "Left"]:
            side_dir = os.path.join(tdir, side)
            if not os.path.isdir(side_dir):
                continue
            for fp in os.listdir(side_dir):
                fp_dir = os.path.join(side_dir, fp)
                if not os.path.isdir(fp_dir):
                    continue
                for f in os.listdir(fp_dir):
                    if f.lower().endswith(".trc"):
                        yield trial_name, side, fp, os.path.join(fp_dir, f)


def main():
    # change
    DATA_ROOT = r"D:\TestOverground\Overground"
    PARTICIPANT = "PLB_03"
    SCALED_MODEL_NAME = "scaledmodelIM.osim"

    participant_root = os.path.join(DATA_ROOT, PARTICIPANT)
    scaled_model_file = os.path.join(participant_root, "models", SCALED_MODEL_NAME)

    processed_root = os.path.join(participant_root, "processed")
    segmented_root = os.path.join(processed_root, "segmented")
    ik_out_root = os.path.join(processed_root, "ik")

    if not os.path.exists(scaled_model_file):
        raise FileNotFoundError(f"Scaled model not found: {scaled_model_file}")
    # # AlterG
    marker_names = [
        'Sternum', 'LShoulder', 'RShoulder', 'LASIS', 'RASIS', 'RPSIS', 'LPSIS',
        'RFibula', 'RShank', 'RAnkleLateral', 'RToe', 'LToe', 'RMT5', 'RMT2', 'RHeel',
        'LFibula', 'LShank', 'LAnkleLateral', 'LMT5', 'LMT2', 'LHeel', 'RKneeLateral',
        'LAnkleMedial', 'LKneeLateral', 'RAnkleMedial', 'LKneeMedial', 'RKneeMedial'
    ]
    do_not_include = ['RKneeMedial', 'RAnkleMedial', 'RToe', 'LKneeMedial', 'LAnkleMedial', 'LToe']

    # ABI full markerset
    # marker_names = [
    #     'CLAV', 'T10', 'C7', 'LACR1', 'LASI', 'LPSI', 'LMFC', 'LLFC', 'LTH1', 'LTH2',
    #     'LTH3', 'LTB1', 'LTB2', 'LTB3', 'LLMAL', 'LMMAL', 'LMT1', 'LMT5', 'LToe', 'LCAL',
    #     'RACR1', 'RASI', 'RPSI', 'RMFC', 'RLFC', 'RTH1', 'RTH2', 'RTH3', 'RTB1', 'RTB2', 'RTB3', 'RLMAL',
    #     'RMMAL', 'RMT1', 'RMT5', 'RToe', 'RCAL'
    # ]
    # do_not_include = ['LMMAL', 'RMMAL', 'RMFC', 'LMFC', 'RToe', 'LToe']

    safe_mkdir(ik_out_root)

    for trial_name, side, fp, trc_path in iter_segmented_trcs(segmented_root):
        trc = TRC.load_from_trc(trc_path)

        cycle_name = os.path.splitext(os.path.basename(trc_path))[0]
        out_dir = os.path.join(ik_out_root, trial_name, side, fp)
        safe_mkdir(out_dir)

        ik_tool = set_up_ik_tool(
            scaled_model_file,
            trc_path,
            float(trc.data["Time"].iloc[0]),
            float(trc.data["Time"].iloc[-1])
        )

        marker_tasks(ik_tool, marker_names, do_not_include)

        mot_path = os.path.join(out_dir, f"{cycle_name}.mot")
        ik_tool.setOutputMotionFileName(mot_path)

        print(f"[IK] Running: {trial_name}/{side}/{fp} -> {os.path.basename(mot_path)}")
        ik_tool.run()

        # Filter and overwrite
        if os.path.exists(mot_path):
            header, time_vec, data = read_mot_storage(mot_path)
            data = filter_signals(data)
            data = np.hstack((time_vec, data))

            mot = MOT.load_from_mot(mot_path)
            df = pd.DataFrame(data)
            df.columns = header
            mot.update_data(df)
            mot.save(out_dir)
        else:
            print(f"[IK] FAILED: {mot_path}")


if __name__ == "__main__":
    main()