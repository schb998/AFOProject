import importlib.util
import os
import sys
import threading
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# -----------------------------------------------------------------------------
# Ensure flet is installed
# -----------------------------------------------------------------------------
ft_spec = importlib.util.find_spec("flet")
if ft_spec is None:
    print("flet not available. Installing...")
    os.system("python -m pip install 'flet[all]'")

import flet as ft

# -----------------------------------------------------------------------------
# Helpers: robust OpenSim .mot/.sto reader + plotting
# -----------------------------------------------------------------------------
def read_opensim_table(path: str):
    """
    Robustly read OpenSim .mot/.sto: locate 'endheader' then parse whitespace table.
    Returns pandas DataFrame with 'time' column.
    """
    import pandas as pd

    with open(path, "r") as f:
        lines = f.readlines()

    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            start = i + 1
            break

    raw = "".join(lines[start:]).strip()
    if not raw:
        raise ValueError(f"No data found after header in {path}")

    df = pd.read_csv(StringIO(raw), sep=r"\s+")
    if "time" not in df.columns and "Time" in df.columns:
        df = df.rename(columns={"Time": "time"})
    if "time" not in df.columns:
        raise KeyError(f"'time' column not found in {path}. Columns={list(df.columns)}")
    return df


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def plot_ik_id_power(
    out_png: str,
    title: str,
    x,
    series: list[tuple[str, object]],
    xlabel="time (s)",
):
    """
    Minimal matplotlib plotter.
    series: list of (label, y-array-like)
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    safe_mkdir(os.path.dirname(out_png))

    plt.figure(figsize=(10, 4))
    for label, y in series:
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Pipeline steps (parameterized, no hardcoded paths)
# -----------------------------------------------------------------------------
def run_postprocessing(data_root: str, participant: str, csv_name: str, threshold: float):
    """
    Calls your existing: OverGround.data_postprocessing.from_app
    """
    import OverGround.data_postprocessing as dp
    dp.from_app({
        "directory": data_root,
        "participant_id": participant,
        "csv_name": csv_name,
        "threshold": threshold
    })


def iter_segmented_trc_paths(segmented_root: str):
    """
    segmented/<Trial>/<Side>/<FPx>/*.trc
    """
    for trial in os.listdir(segmented_root):
        tdir = os.path.join(segmented_root, trial)
        if not os.path.isdir(tdir):
            continue
        for side in ["Left", "Right"]:
            sdir = os.path.join(tdir, side)
            if not os.path.isdir(sdir):
                continue
            for fp in os.listdir(sdir):
                fpdir = os.path.join(sdir, fp)
                if not os.path.isdir(fpdir):
                    continue
                for f in os.listdir(fpdir):
                    if f.lower().endswith(".trc"):
                        yield trial, side, fp, os.path.join(fpdir, f)


def run_ik_overground(data_root: str, participant: str, scaled_model_name: str, make_plots: bool):
    """
    Runs IK for every segmented TRC.
    Uses your same OpenSim IK logic but parameterized.
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import butter, filtfilt
    import opensim as osim

    # Allow importing resources/*
    # Assumes this GUI file lives in the same project root (or adjust as needed)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from resources.file_types.mot import MOT
    from resources.file_types.trc import TRC

    participant_root = os.path.join(data_root, participant)
    scaled_model_file = os.path.join(participant_root, "models", scaled_model_name)

    processed_root = os.path.join(participant_root, "processed")
    segmented_root = os.path.join(processed_root, "segmented")
    ik_out_root = os.path.join(processed_root, "ik")
    plots_root = os.path.join(processed_root, "plots", "ik")

    if not os.path.exists(scaled_model_file):
        raise FileNotFoundError(f"Scaled model not found: {scaled_model_file}")
    if not os.path.isdir(segmented_root):
        raise FileNotFoundError(f"Segmented folder not found: {segmented_root}")

    safe_mkdir(ik_out_root)

    # marker set (same as your script)
    marker_names = [
        'Sternum', 'LShoulder', 'RShoulder', 'LASIS', 'RASIS', 'RPSIS', 'LPSIS',
        'RFibula', 'RShank', 'RAnkleLateral', 'RToe', 'LToe', 'RMT5', 'RMT2', 'RHeel',
        'LFibula', 'LShank', 'LAnkleLateral', 'LMT5', 'LMT2', 'LHeel', 'RKneeLateral',
        'LAnkleMedial', 'LKneeLateral', 'RAnkleMedial', 'LKneeMedial', 'RKneeMedial'
    ]
    do_not_include = ['RKneeMedial', 'RAnkleMedial', 'RToe', 'LKneeMedial', 'LAnkleMedial', 'LToe']

    def filter_signals(data: np.ndarray, fs: float, cutoff: float = 6.0, order: int = 2) -> np.ndarray:
        nyq = 0.5 * fs
        if cutoff >= nyq:
            return data
        b, a = butter(order, cutoff / nyq, btype="low", analog=False)
        return filtfilt(b, a, data, axis=0)

    def read_mot_storage(filepath: str):
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

    def set_up_ik_tool(model_file, marker_data, start_time, end_time):
        tool = osim.InverseKinematicsTool()
        tool.set_model_file(model_file)
        tool.setMarkerDataFileName(marker_data)
        tool.setStartTime(start_time)
        tool.setEndTime(end_time)
        return tool

    def marker_tasks(tool, markers, do_not_include_list):
        taskset = tool.getIKTaskSet()
        for m in markers:
            task = osim.IKMarkerTask()
            task.setName(m)
            task.setApply(m not in do_not_include_list)
            task.setWeight(1)
            taskset.cloneAndAppend(task)

    for trial, side, fp, trc_path in iter_segmented_trc_paths(segmented_root):
        trc = TRC.load_from_trc(trc_path)
        cycle_name = os.path.splitext(os.path.basename(trc_path))[0]

        out_dir = os.path.join(ik_out_root, trial, side, fp)
        safe_mkdir(out_dir)

        start_time = float(trc.data["Time"].iloc[0])
        end_time = float(trc.data["Time"].iloc[-1])

        ik_tool = set_up_ik_tool(scaled_model_file, trc_path, start_time, end_time)
        marker_tasks(ik_tool, marker_names, do_not_include)

        # Save OpenSim output without overwriting (more robust)
        mot_raw_path = os.path.join(out_dir, f"{cycle_name}_ik_raw.mot")
        ik_tool.setOutputMotionFileName(mot_raw_path)

        print(f"[IK] Running: {trial}/{side}/{fp} -> {os.path.basename(mot_raw_path)}")
        ik_tool.run()

        # Optionally also create a filtered copy (but do NOT overwrite raw)
        if os.path.exists(mot_raw_path):
            header, time_vec, data = read_mot_storage(mot_raw_path)
            fs = 1.0 / float(np.mean(np.diff(time_vec[:, 0])))
            data_f = filter_signals(data, fs=fs, cutoff=6.0, order=2)
            combined = np.hstack((time_vec, data_f))

            mot = MOT.load_from_mot(mot_raw_path)
            df = pd.DataFrame(combined)
            df.columns = header
            mot.update_data(df)

            mot_filt_path = os.path.join(out_dir, f"{cycle_name}_ik_filt.mot")
            mot.rename(filename=os.path.basename(mot_filt_path))
            mot.save(out_dir)

            if make_plots:
                # quick plot for ankle angle if present
                try:
                    dfp = read_opensim_table(mot_filt_path)
                    col = "ankle_angle_l" if side == "Left" else "ankle_angle_r"
                    if col in dfp.columns:
                        out_png = os.path.join(plots_root, trial, side, fp, f"{cycle_name}_ik.png")
                        plot_ik_id_power(
                            out_png=out_png,
                            title=f"{trial} {side} {fp} IK",
                            x=dfp["time"].to_numpy(),
                            series=[(col, dfp[col].to_numpy())],
                        )
                except Exception as e:
                    print(f"[IK] Plot skipped ({cycle_name}): {repr(e)}")
        else:
            print(f"[IK] FAILED: {mot_raw_path}")


def iter_segmented_grf_cycles(segmented_root: str):
    """
    segmented/<Trial>/<Side>/<FPx>/*.mot  (GRF)
    """
    for trial in os.listdir(segmented_root):
        tdir = os.path.join(segmented_root, trial)
        if not os.path.isdir(tdir):
            continue
        for side in ["Left", "Right"]:
            sdir = os.path.join(tdir, side)
            if not os.path.isdir(sdir):
                continue
            for fp in os.listdir(sdir):
                fpdir = os.path.join(sdir, fp)
                if not os.path.isdir(fpdir):
                    continue
                for f in os.listdir(fpdir):
                    if f.lower().endswith(".mot"):
                        yield trial, side, fp, os.path.join(fpdir, f)


def run_id_overground(data_root: str, participant: str, scaled_model_name: str, make_plots: bool):
    """
    Runs ID for each segmented GRF cycle.
    Robust ExternalLoads: ONLY uses the plate implied by folder FPx, and only if non-zero.
    Uses TRC time as master, but clamps to overlap with GRF time.
    """
    import opensim as osim
    import pandas as pd

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from resources.file_types.mot import MOT
    from resources.file_types.trc import TRC

    participant_root = os.path.join(data_root, participant)
    model_file = os.path.join(participant_root, "models", scaled_model_name)
    processed_root = os.path.join(participant_root, "processed")

    segmented_root = os.path.join(processed_root, "segmented")
    ik_root = os.path.join(processed_root, "ik")
    id_root = os.path.join(processed_root, "id")
    exloads_root = os.path.join(processed_root, "external_loads")
    infosheet_dir = os.path.join(participant_root, "infosheet")
    plots_root = os.path.join(processed_root, "plots", "id")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Scaled model not found: {model_file}")
    if not os.path.isdir(segmented_root):
        raise FileNotFoundError(f"Segmented folder not found: {segmented_root}")
    if not os.path.isdir(ik_root):
        raise FileNotFoundError(f"IK folder not found: {ik_root}")

    # --- infosheet reader (lightweight) ---
    def find_first_csv(folder: str) -> str:
        csvs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No infosheet CSV in: {folder}")
        csvs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return csvs[0]

    info_csv = find_first_csv(infosheet_dir)
    info = pd.read_csv(info_csv)

    def row_for_trial(trial: str):
        m = info[info["Trials/Events"].astype(str).str.lower() == str(trial).lower()]
        return None if m.empty else m.iloc[0]

    def parse_role(x: str) -> str:
        s = str(x).strip().lower()
        if s in ("left", "l"):
            return "left"
        if s in ("right", "r"):
            return "right"
        if s in ("both", "b"):
            return "both"
        return "none"

    def parse_plate_num(fp_folder: str):
        s = fp_folder.strip().upper()
        if not s.startswith("FP"):
            return None
        try:
            return int(s.replace("FP", "").strip())
        except ValueError:
            return None

    def plate_allowed(trial: str, plate_num: int, side: str) -> bool:
        """
        Enforce your rule:
          - if infosheet says this plate is Both -> do NOT include it in XML
          - if it says Left/Right -> must match cycle side
        """
        r = row_for_trial(trial)
        if r is None:
            # if no infosheet row, be conservative: allow matching side only
            return True

        cell = r.get(f"FP{plate_num}", "")
        role = parse_role(cell)
        if role == "both":
            return False
        if role == "left" and side.lower() == "left":
            return True
        if role == "right" and side.lower() == "right":
            return True
        # unknown/none mismatch
        return False

    def nonzero_plate(df, plate_num: int) -> bool:
        cols = [f"ground_force{plate_num}_vx", f"ground_force{plate_num}_vy", f"ground_force{plate_num}_vz"]
        for c in cols:
            if c not in df.columns:
                return False
        return float(df[cols].abs().sum().sum()) > 0.0

    def build_external_loads_xml(grf_mot_path: str, xml_path: str, side: str, plate_num: int):
        mot = MOT.load_from_mot(grf_mot_path)
        df = mot.data

        if not plate_allowed(trial_name, plate_num, side):
            print(f"[EXLOADS] Skip XML: infosheet says FP{plate_num} is Both or mismatch for {trial_name}/{side}")
            return None

        if not nonzero_plate(df, plate_num):
            print(f"[EXLOADS] Skip XML: FP{plate_num} force columns are zero in {os.path.basename(grf_mot_path)}")
            return None

        ext_loads = osim.ExternalLoads()
        ext_loads.setDataFileName(grf_mot_path)

        ext = osim.ExternalForce()
        ext.setName(f"FP{plate_num}_{side}")
        ext.set_applied_to_body("calcn_l" if side.lower() == "left" else "calcn_r")
        ext.set_force_expressed_in_body("ground")
        ext.set_point_expressed_in_body("ground")
        ext.set_force_identifier(f"ground_force{plate_num}_v")
        ext.set_point_identifier(f"ground_force{plate_num}_p")
        ext.set_torque_identifier(f"ground_torque{plate_num}_")

        ext_loads.cloneAndAppend(ext)

        safe_mkdir(os.path.dirname(xml_path))
        ext_loads.printToXML(xml_path)
        print(f"[EXLOADS] Wrote: {xml_path}")
        return xml_path

    def get_time_range_from_trc(trc_path: str):
        trc = TRC.load_from_trc(trc_path)
        t0 = float(trc.data["Time"].iloc[0])
        t1 = float(trc.data["Time"].iloc[-1])
        return t0, t1

    def get_time_range_from_grf(grf_mot_path: str):
        mot = MOT.load_from_mot(grf_mot_path)
        if "time" not in mot.data.columns:
            raise KeyError(f"'time' column not found in: {grf_mot_path}")
        return float(mot.data["time"].iloc[0]), float(mot.data["time"].iloc[-1])

    def setup_id_tool(model_file: str, start_time: float, end_time: float, ik_path: str, xml_path: str, out_path: str):
        tool = osim.InverseDynamicsTool()
        tool.setModelFileName(model_file)
        tool.setStartTime(start_time)
        tool.setEndTime(end_time)
        tool.setCoordinatesFileName(ik_path)
        tool.setExternalLoadsFileName(xml_path)
        tool.setResultsDir(os.path.dirname(out_path))
        tool.setOutputGenForceFileName(os.path.basename(out_path))
        return tool

    for trial_name, side, fp, grf_path in iter_segmented_grf_cycles(segmented_root):
        plate_num = parse_plate_num(fp)
        if plate_num is None:
            print(f"[ID] Skip: can't parse plate number from '{fp}'")
            continue

        cycle_name = os.path.splitext(os.path.basename(grf_path))[0]

        # matching TRC:
        trc_path = os.path.join(segmented_root, trial_name, side, fp, f"{cycle_name}.trc")
        if not os.path.exists(trc_path):
            print(f"[ID] Missing TRC: {trc_path}")
            continue

        # matching IK: prefer filtered if exists, else raw
        ik_dir = os.path.join(ik_root, trial_name, side, fp)
        ik_filt = os.path.join(ik_dir, f"{cycle_name}_ik_filt.mot")
        ik_raw = os.path.join(ik_dir, f"{cycle_name}_ik_raw.mot")
        ik_path = ik_filt if os.path.exists(ik_filt) else ik_raw
        if not os.path.exists(ik_path):
            print(f"[ID] Missing IK: {ik_path}")
            continue

        # master window from TRC, clamp to GRF overlap
        trc_t0, trc_t1 = get_time_range_from_trc(trc_path)
        grf_t0, grf_t1 = get_time_range_from_grf(grf_path)
        t0 = max(trc_t0, grf_t0)
        t1 = min(trc_t1, grf_t1)
        if t1 <= t0:
            print(f"[ID] Skip: no overlap TRC vs GRF for {trial_name}/{side}/{fp}/{cycle_name}")
            continue

        xml_path = os.path.join(exloads_root, trial_name, side, fp, f"{cycle_name}.xml")
        xml_ok = build_external_loads_xml(grf_path, xml_path, side, plate_num)
        if xml_ok is None:
            continue

        out_dir = os.path.join(id_root, trial_name, side, fp)
        safe_mkdir(out_dir)
        out_path = os.path.join(out_dir, f"{cycle_name}_id.mot")

        print(f"[ID] Running: {trial_name}/{side}/{fp}/{cycle_name}")
        tool = setup_id_tool(model_file, t0, t1, ik_path, xml_ok, out_path)

        try:
            tool.run()
        except Exception as e:
            print(f"[ID] FAILED: {trial_name}/{side}/{fp}/{cycle_name} -> {repr(e)}")
            continue

        if os.path.exists(out_path) and make_plots:
            try:
                dfid = read_opensim_table(out_path)
                # plot ankle moment if present
                col = "ankle_angle_l_moment" if side == "Left" else "ankle_angle_r_moment"
                if col in dfid.columns:
                    out_png = os.path.join(plots_root, trial_name, side, fp, f"{cycle_name}_id.png")
                    plot_ik_id_power(
                        out_png=out_png,
                        title=f"{trial_name} {side} {fp} ID",
                        x=dfid["time"].to_numpy(),
                        series=[(col, dfid[col].to_numpy())],
                    )
            except Exception as e:
                print(f"[ID] Plot skipped ({cycle_name}): {repr(e)}")


def run_power_overground(data_root: str, participant: str, make_plots: bool):
    """
    Calls your joint_power_overground.py main logic by importing and running a parameterized wrapper.
    Assumes:
      processed/ik/<trial>/<side>/<FPx>/*_ik_filt.mot (preferred) or *_ik_raw.mot
      processed/id/<trial>/<side>/<FPx>/*_id.mot
    Saves:
      processed/power/<trial>/<side>/<FPx>/*_power_time.csv and *_power_gc.csv
    Also saves plots if make_plots=True.
    """
    import numpy as np
    import pandas as pd

    participant_root = os.path.join(data_root, participant)
    processed_root = os.path.join(participant_root, "processed")
    ik_root = os.path.join(processed_root, "ik")
    id_root = os.path.join(processed_root, "id")
    power_root = os.path.join(processed_root, "power")
    plots_root = os.path.join(processed_root, "plots", "power")

    safe_mkdir(power_root)

    # --- helper: find matching ID file for a cycle ---
    def find_id_for_cycle(trial, side, fp, cycle_base):
        cand = os.path.join(id_root, trial, side, fp, f"{cycle_base}_id.mot")
        return cand if os.path.exists(cand) else None

    def iter_ik_cycles():
        for trial in os.listdir(ik_root):
            tdir = os.path.join(ik_root, trial)
            if not os.path.isdir(tdir):
                continue
            for side in ["Left", "Right"]:
                sdir = os.path.join(tdir, side)
                if not os.path.isdir(sdir):
                    continue
                for fp in os.listdir(sdir):
                    fpdir = os.path.join(sdir, fp)
                    if not os.path.isdir(fpdir):
                        continue
                    for f in os.listdir(fpdir):
                        fl = f.lower()
                        if fl.endswith("_ik_filt.mot") or fl.endswith("_ik_raw.mot"):
                            yield trial, side, fp, os.path.join(fpdir, f)

    def time_normalize(df: pd.DataFrame, xcol="time", n=101) -> pd.DataFrame:
        x = df[xcol].to_numpy()
        if len(x) < 2:
            return df.copy()
        x0, x1 = float(x[0]), float(x[-1])
        if x1 <= x0:
            return df.copy()

        gc = np.linspace(0.0, 100.0, n)
        new = {"gc": gc}

        for c in df.columns:
            if c == xcol:
                continue
            y = df[c].to_numpy()
            new[c] = np.interp(gc, np.linspace(0.0, 100.0, len(y)), y)

        return pd.DataFrame(new)

    for trial, side, fp, ik_path in iter_ik_cycles():
        ik_df = read_opensim_table(ik_path)

        cycle_base = os.path.basename(ik_path).replace("_ik_filt.mot", "").replace("_ik_raw.mot", "")
        id_path = find_id_for_cycle(trial, side, fp, cycle_base)
        if id_path is None:
            print(f"[POWER] Missing ID for {trial}/{side}/{fp}/{cycle_base}")
            continue

        id_df = read_opensim_table(id_path)

        # Interpolate ID to IK time
        t = ik_df["time"].to_numpy()
        idt = id_df["time"].to_numpy()

        # Moments columns end with _moment in your ID outputs
        moment_cols = [c for c in id_df.columns if c.endswith("_moment")]
        if not moment_cols:
            print(f"[POWER] No *_moment columns in: {id_path}")
            continue

        id_interp = {"time": t}
        for c in moment_cols:
            id_interp[c] = np.interp(t, idt, id_df[c].to_numpy())

        idI = pd.DataFrame(id_interp)

        # Angular velocity from IK in rad/s (finite diff)
        # Choose angle columns consistent with side
        if side == "Left":
            angle_cols = [c for c in ik_df.columns if c.endswith("_l")]
        else:
            angle_cols = [c for c in ik_df.columns if c.endswith("_r")]

        # Use ankle_angle_{l/r} if present, plus pelvis as you like
        # Compute omega for those with matching moment columns
        power = {"time": t}

        dt = np.gradient(t)
        dt[dt == 0] = np.nan

        for ang in angle_cols:
            base = ang
            mom = f"{base}_moment"
            if mom not in idI.columns:
                continue
            # convert deg to rad then diff
            theta = np.deg2rad(ik_df[ang].to_numpy())
            omega = np.gradient(theta) / dt
            omega = np.nan_to_num(omega)

            pcol = f"{base}_power"
            power[pcol] = omega * idI[mom].to_numpy()

        power_df = pd.DataFrame(power)

        out_dir = os.path.join(power_root, trial, side, fp)
        safe_mkdir(out_dir)

        out_time_csv = os.path.join(out_dir, f"{cycle_base}_power_time.csv")
        power_df.to_csv(out_time_csv, index=False)

        out_gc_csv = os.path.join(out_dir, f"{cycle_base}_power_gc.csv")
        power_gc = time_normalize(power_df, xcol="time", n=101)
        power_gc.to_csv(out_gc_csv, index=False)

        if make_plots:
            # plot ankle power if exists
            target = "ankle_angle_l_power" if side == "Left" else "ankle_angle_r_power"
            if target in power_df.columns:
                out_png = os.path.join(plots_root, trial, side, fp, f"{cycle_base}_power_time.png")
                plot_ik_id_power(
                    out_png=out_png,
                    title=f"{trial} {side} {fp} Power (time)",
                    x=power_df["time"].to_numpy(),
                    series=[(target, power_df[target].to_numpy())],
                )
            if target in power_gc.columns:
                out_png = os.path.join(plots_root, trial, side, fp, f"{cycle_base}_power_gc.png")
                plot_ik_id_power(
                    out_png=out_png,
                    title=f"{trial} {side} {fp} Power (0–100%)",
                    x=power_gc["gc"].to_numpy(),
                    series=[(target, power_gc[target].to_numpy())],
                    xlabel="gait cycle (%)",
                )

        print(f"[POWER] Saved: {out_time_csv}")
        print(f"[POWER] Saved: {out_gc_csv}")


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
def main_app(page: ft.Page):
    page.title = "Overground Pipeline Runner"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 740
    page.window.height = 900

    # Inputs
    data_root = ft.TextField(label="DATA_ROOT", value=r"D:\TestOverground\Overground", width=620)
    participant = ft.TextField(label="Participant ID", value="PLB_03", width=300)
    csv_name = ft.TextField(label="Infosheet CSV Name", value="Trials_PLB_03.csv", width=300)
    threshold = ft.TextField(label="Contact Threshold (N)", value="20.0", width=300)
    model_name = ft.TextField(label="Scaled model filename", value="scaledmodelIM.osim", width=300)

    # Options
    run_post = ft.Checkbox(label="Run postprocessing (GRF correction + segmentation)", value=True)
    run_ik = ft.Checkbox(label="Run IK (from segmented TRC)", value=True)
    run_id = ft.Checkbox(label="Run ID (TRC master window + robust ExternalLoads)", value=True)
    run_power = ft.Checkbox(label="Run Power (time + 0–100% GC)", value=True)
    make_plots = ft.Checkbox(label="Save plots (IK/ID/Power)", value=True)

    # Logging
    log_box = ft.TextField(
        label="Log",
        multiline=True,
        min_lines=18,
        max_lines=28,
        read_only=True,
        width=700,
    )

    def log(msg: str):
        log_box.value += msg + "\n"
        log_box.update()

    def run_in_thread():
        buff = StringIO()
        try:
            with redirect_stdout(buff), redirect_stderr(buff):
                dr = data_root.value.strip()
                pid = participant.value.strip()
                csvn = csv_name.value.strip()
                thr = float(threshold.value.strip())
                mdl = model_name.value.strip()
                plots = bool(make_plots.value)

                # 1) Post
                if run_post.value:
                    print("\n=== POSTPROCESSING ===")
                    run_postprocessing(dr, pid, csvn, thr)

                # 2) IK
                if run_ik.value:
                    print("\n=== IK ===")
                    run_ik_overground(dr, pid, mdl, plots)

                # 3) ID
                if run_id.value:
                    print("\n=== ID ===")
                    run_id_overground(dr, pid, mdl, plots)

                # 4) Power
                if run_power.value:
                    print("\n=== POWER ===")
                    run_power_overground(dr, pid, plots)

                print("\n[Done] Pipeline completed.")
        except Exception:
            buff.write("\n[ERROR]\n")
            buff.write(traceback.format_exc())
        finally:
            # dump all output into log box
            out = buff.getvalue()
            page.call_from_thread(lambda: log(out))

    def on_run_all(e):
        log_box.value = ""
        log_box.update()
        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()
        log("Started pipeline...\n")

    run_btn = ft.ElevatedButton(text="Run pipeline", on_click=on_run_all, width=220)

    page.add(
        ft.Text("Overground Full Pipeline Runner", size=26, weight="bold"),
        ft.Row([data_root]),
        ft.Row([participant, csv_name]),
        ft.Row([threshold, model_name]),
        ft.Divider(),
        run_post,
        run_ik,
        run_id,
        run_power,
        make_plots,
        ft.Divider(),
        run_btn,
        log_box,
    )


if __name__ == "__main__":
    ft.run(main_app)
