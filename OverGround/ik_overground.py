import os
import pathlib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import opensim as osim

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC


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
    Expects structure produced by data_postprocessing.py:
      segmented/Right/FP1/cycle_1/*.trc
      segmented/Left/FP2/cycle_3/*.trc
      ...
    Yields (side, fp_folder, cycle_folder, trc_path)
    """
    for side in ["Right", "Left"]:
        side_dir = os.path.join(segmented_root, side)
        if not os.path.isdir(side_dir):
            continue
        for fp in os.listdir(side_dir):
            fp_dir = os.path.join(side_dir, fp)
            if not os.path.isdir(fp_dir):
                continue
            for cycle_folder in os.listdir(fp_dir):
                cdir = os.path.join(fp_dir, cycle_folder)
                if not os.path.isdir(cdir):
                    continue
                trcs = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(".trc")]
                for trc_path in trcs:
                    yield side, fp, cycle_folder, trc_path


def main():
    # ===================== EDIT THESE =====================
    DATA_ROOT = r"D:\TestOverground\Overground"
    PARTICIPANT = "PLB_02"
    SCALED_MODEL_NAME = "scaled_model.osim"  # in participant_root\models\
    # ======================================================

    participant_root = os.path.join(DATA_ROOT, PARTICIPANT)
    scaled_model_file = os.path.join(participant_root, "models", SCALED_MODEL_NAME)

    processed_root = os.path.join(participant_root, "processed")
    segmented_root = os.path.join(processed_root, "segmented")
    ik_out_root = os.path.join(processed_root, "ik")

    if not os.path.exists(scaled_model_file):
        raise FileNotFoundError(f"Scaled model not found: {scaled_model_file}")

    marker_names = [
        'Sternum', 'LShoulder', 'RShoulder', 'LASIS', 'RASIS', 'RPSIS', 'LPSIS',
        'RFibula', 'RShank', 'RAnkleLateral', 'RToe', 'LToe', 'RMT5', 'RMT2', 'RHeel',
        'LFibula', 'LShank', 'LAnkleLateral', 'LMT5', 'LMT2', 'LHeel', 'RKneeLateral',
        'LAnkleMedial', 'LKneeLateral', 'RAnkleMedial', 'LKneeMedial', 'RKneeMedial'
    ]
    do_not_include = ['RKneeMedial', 'RAnkleMedial', 'RToe', 'LKneeMedial', 'LAnkleMedial', 'LToe']

    safe_mkdir(ik_out_root)

    for side, fp, cycle_folder, trc_path in iter_segmented_trcs(segmented_root):
        trc = TRC.load_from_trc(trc_path)

        cycle_name = os.path.splitext(os.path.basename(trc_path))[0]  # e.g. Walk01-02_Right_cycle1
        out_dir = os.path.join(ik_out_root, side, fp, cycle_folder)
        safe_mkdir(out_dir)

        # Run IK
        ik_tool = set_up_ik_tool(
            scaled_model_file,
            trc_path,
            float(trc.data["Time"].iloc[0]),
            float(trc.data["Time"].iloc[-1])
        )
        marker_tasks(ik_tool, marker_names, do_not_include)

        mot_path = os.path.join(out_dir, f"{cycle_name}.mot")
        ik_tool.setOutputMotionFileName(mot_path)

        print(f"[IK] Running: {side}/{fp}/{cycle_folder} -> {os.path.basename(mot_path)}")
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

    print("\n[Done] IK completed.")


if __name__ == "__main__":
    main()