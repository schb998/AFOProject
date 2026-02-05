import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opensim as osim

# Add project root to path (so `resources` can be imported when run as a script)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resources.file_types.trc import TRC


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def read_manifest(manifest_path: str) -> pd.DataFrame | None:
    if not os.path.exists(manifest_path):
        return None
    df = pd.read_csv(manifest_path)
    # expected columns from your segmentation script
    needed = {"trial", "side", "start_plate", "trc_path"}
    if not needed.issubset(set(df.columns)):
        print(f"[IK] Manifest missing columns {needed - set(df.columns)}. Falling back to directory scan.")
        return None
    return df


def iter_cycles_from_manifest(df: pd.DataFrame):
    for _, r in df.iterrows():
        trial = str(r["trial"])
        side = str(r["side"])
        plate = int(r["start_plate"])
        trc_path = str(r["trc_path"])
        if not os.path.exists(trc_path):
            print(f"[IK] Skip missing TRC: {trc_path}")
            continue
        yield trial, side, f"FP{plate}", trc_path


def iter_segmented_trcs(segmented_root: str):
    """
    Fallback: segmented/<Trial>/<Side>/<FPx>/*.trc
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


def set_up_ik_tool(model_file: str, marker_data: str, start_time: float, end_time: float) -> osim.InverseKinematicsTool:
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


def plot_ik(mot_path: str, out_png: str, side: str):
    """
    Minimal diagnostic plot: ankle angle + knee angle (if present).
    """
    try:
        import pandas as pd
        # openSim .mot sometimes has varying header length; find endheader
        with open(mot_path, "r") as f:
            lines = f.readlines()
        start = 0
        for i, line in enumerate(lines):
            if line.strip().lower() == "endheader":
                start = i + 1
                break
        data = pd.read_csv(pd.io.common.StringIO("".join(lines[start:])), sep=r"\s+")
        if "time" not in data.columns and "Time" in data.columns:
            data = data.rename(columns={"Time": "time"})
        t = data["time"].to_numpy()
    except Exception as e:
        print(f"[IK] Plot skip (cannot read mot): {mot_path} -> {repr(e)}")
        return

    ankle = "ankle_angle_r" if side == "Right" else "ankle_angle_l"
    knee = "knee_angle_r" if side == "Right" else "knee_angle_l"

    plt.figure(figsize=(10, 4))
    if ankle in data.columns:
        plt.plot(t, data[ankle], label=ankle)
    if knee in data.columns:
        plt.plot(t, data[knee], label=knee)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(os.path.basename(mot_path))
    plt.grid(True)
    plt.legend()
    safe_mkdir(os.path.dirname(out_png))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    DATA_ROOT = r"D:\TestOverground\Overground"
    PARTICIPANT = "PLB_03"
    SCALED_MODEL_NAME = "scaledmodelIM.osim"

    participant_root = os.path.join(DATA_ROOT, PARTICIPANT)
    scaled_model_file = os.path.join(participant_root, "models", SCALED_MODEL_NAME)

    processed_root = os.path.join(participant_root, "processed")
    segmented_root = os.path.join(processed_root, "segmented")
    ik_out_root = os.path.join(processed_root, "ik")
    plots_root = os.path.join(processed_root, "plots", "ik")
    manifests_root = os.path.join(processed_root, "manifests")
    manifest_path = os.path.join(manifests_root, "overground_cycles_manifest.csv")

    if not os.path.exists(scaled_model_file):
        raise FileNotFoundError(f"Scaled model not found: {scaled_model_file}")

    safe_mkdir(ik_out_root)

    # Marker set (your AlterG list)
    marker_names = [
        'Sternum', 'LShoulder', 'RShoulder', 'LASIS', 'RASIS', 'RPSIS', 'LPSIS',
        'RFibula', 'RShank', 'RAnkleLateral', 'RToe', 'LToe', 'RMT5', 'RMT2', 'RHeel',
        'LFibula', 'LShank', 'LAnkleLateral', 'LMT5', 'LMT2', 'LHeel', 'RKneeLateral',
        'LAnkleMedial', 'LKneeLateral', 'RAnkleMedial', 'LKneeMedial', 'RKneeMedial'
    ]
    do_not_include = ['RKneeMedial', 'RAnkleMedial', 'RToe', 'LKneeMedial', 'LAnkleMedial', 'LToe']

    df = read_manifest(manifest_path)
    if df is not None:
        iterator = iter_cycles_from_manifest(df)
        print(f"[IK] Using manifest: {manifest_path}")
    else:
        iterator = iter_segmented_trcs(segmented_root)
        print(f"[IK] Using directory scan: {segmented_root}")

    n_ok = 0
    n_fail = 0

    for trial_name, side, fp, trc_path in iterator:
        trc = TRC.load_from_trc(trc_path)
        if trc.data.shape[0] < 2:
            print(f"[IK] Skip: TRC has no data rows: {trc_path}")
            n_fail += 1
            continue

        cycle_name = os.path.splitext(os.path.basename(trc_path))[0]
        out_dir = os.path.join(ik_out_root, trial_name, side, fp)
        safe_mkdir(out_dir)

        start_time = float(trc.data["Time"].iloc[0])
        end_time = float(trc.data["Time"].iloc[-1])

        ik_tool = set_up_ik_tool(scaled_model_file, trc_path, start_time, end_time)
        marker_tasks(ik_tool, marker_names, do_not_include)

        mot_raw = os.path.join(out_dir, f"{cycle_name}_ik_raw.mot")
        ik_tool.setOutputMotionFileName(mot_raw)

        print(f"[IK] Running: {trial_name}/{side}/{fp} -> {os.path.basename(mot_raw)}")
        try:
            ik_tool.run()
        except Exception as e:
            print(f"[IK] FAILED: {trc_path} -> {repr(e)}")
            n_fail += 1
            continue

        if os.path.exists(mot_raw):
            n_ok += 1
            # Plot (diagnostic)
            out_png = os.path.join(plots_root, trial_name, side, fp, f"{cycle_name}_ik.png")
            plot_ik(mot_raw, out_png, side)
        else:
            print(f"[IK] FAILED: output not found: {mot_raw}")
            n_fail += 1

    print(f"\n[Done] IK completed. OK={n_ok}, FAIL={n_fail}")


if __name__ == "__main__":
    main()
