import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opensim as osim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


# -------------------- infosheet --------------------

def parse_side_cell(x: str) -> str:
    val = str(x).strip().lower()
    if val in ("l", "left"):
        return "left"
    if val in ("r", "right"):
        return "right"
    if val in ("b", "both"):
        return "both"
    return "none"


class OvergroundInfoSheet:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path).copy()
        # normalize
        for c in self.df.columns:
            if self.df[c].dtype == object:
                self.df[c] = self.df[c].astype(str).str.strip()

    def row_for_trial(self, trial_name: str) -> pd.Series:
        rows = self.df[self.df["Trials/Events"].astype(str).str.lower() == str(trial_name).lower()]
        if rows.empty:
            raise ValueError(f"Trial '{trial_name}' not found in infosheet.")
        return rows.iloc[0]

    def fp_role(self, trial_name: str, plate_num: int) -> str:
        r = self.row_for_trial(trial_name)
        return parse_side_cell(r.get(f"FP{plate_num}", ""))


# -------------------- helpers --------------------

def read_manifest(manifest_path: str) -> pd.DataFrame | None:
    if not os.path.exists(manifest_path):
        return None
    df = pd.read_csv(manifest_path)
    needed = {"trial", "side", "start_plate", "grf_path", "trc_path"}
    if not needed.issubset(set(df.columns)):
        print(f"[ID] Manifest missing columns {needed - set(df.columns)}. Falling back to directory scan.")
        return None
    return df


def iter_cycles_from_manifest(df: pd.DataFrame):
    for _, r in df.iterrows():
        trial = str(r["trial"])
        side = str(r["side"])
        plate = int(r["start_plate"])
        grf_path = str(r["grf_path"])
        trc_path = str(r["trc_path"])
        if not os.path.exists(grf_path):
            print(f"[ID] Skip missing GRF: {grf_path}")
            continue
        if not os.path.exists(trc_path):
            print(f"[ID] Skip missing TRC: {trc_path}")
            continue
        yield trial, side, plate, grf_path, trc_path


def iter_segmented_grf(segmented_root: str):
    """
    segmented/<Trial>/<Side>/<FPx>/*.mot
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
                        grf_path = os.path.join(fpdir, f)
                        trc_path = os.path.splitext(grf_path)[0] + ".trc"
                        try:
                            plate = int(fp.replace("FP", "").strip())
                        except ValueError:
                            continue
                        if os.path.exists(trc_path):
                            yield trial, side, plate, grf_path, trc_path


def get_time_range_from_grf(grf_mot_path: str):
    mot = MOT.load_from_mot(grf_mot_path)
    if "time" not in mot.data.columns:
        raise KeyError(f"'time' not found in GRF mot: {grf_mot_path}")
    return float(mot.data["time"].iloc[0]), float(mot.data["time"].iloc[-1])


def get_time_range_from_trc(trc_path: str):
    trc = TRC.load_from_trc(trc_path)
    if "Time" not in trc.data.columns:
        raise KeyError(f"'Time' not found in TRC: {trc_path}")
    if trc.data.shape[0] < 2:
        raise IndexError("TRC has <2 rows")
    return float(trc.data["Time"].iloc[0]), float(trc.data["Time"].iloc[-1])


def plate_nonzero(df: pd.DataFrame, plate_num: int) -> bool:
    cols = [f"ground_force{plate_num}_vx", f"ground_force{plate_num}_vy", f"ground_force{plate_num}_vz"]
    if not all(c in df.columns for c in cols):
        return False
    return float(df[cols].abs().sum().sum()) > 0.0


def peak_vy(df: pd.DataFrame, plate_num: int) -> float:
    col = f"ground_force{plate_num}_vy"
    if col not in df.columns:
        return 0.0
    return float(df[col].abs().max())


def build_external_loads_xml_single_plate(
    grf_mot_path: str,
    xml_path: str,
    trial: str,
    side: str,
    plate_num: int,
    infosheet: OvergroundInfoSheet,
    min_peak_vy_n: float = 20.0,
):
    """
    Robust overground rule:
      - ONLY include the cycle folder plate (plate_num)
      - ONLY if force columns are non-zero
      - Skip if infosheet says that plate is 'Both'
    """
    mot = MOT.load_from_mot(grf_mot_path)
    df = mot.data

    role = infosheet.fp_role(trial, plate_num)  # left/right/both/none

    if role == "both":
        print(f"[EXLOADS] Skip XML: infosheet FP{plate_num}=Both for {trial} (per rule).")
        return None

    # optional sanity: skip if side mismatch with infosheet role
    if role in ("left", "right"):
        expected = "Left" if role == "left" else "Right"
        if expected.lower() != side.lower():
            print(f"[EXLOADS] Skip XML: side mismatch {trial} cycle side={side} but FP{plate_num}={expected} in infosheet.")
            return None

    if not plate_nonzero(df, plate_num):
        print(f"[EXLOADS] Skip XML: FP{plate_num} forces are all zero: {os.path.basename(grf_mot_path)}")
        return None

    pk = peak_vy(df, plate_num)
    if pk < min_peak_vy_n:
        print(f"[EXLOADS] Skip XML: FP{plate_num} peak vy < {min_peak_vy_n} N: {os.path.basename(grf_mot_path)}")
        return None

    applied_body = "calcn_l" if side.lower() == "left" else "calcn_r"

    ext_loads = osim.ExternalLoads()
    ext_loads.setDataFileName(grf_mot_path)

    ext = osim.ExternalForce()
    ext.setName(f"FP{plate_num}_{side}")
    ext.set_applied_to_body(applied_body)
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


def setup_id_tool(model_file: str, start_time: float, end_time: float, ik_path: str, xml_path: str, output_mot_path: str):
    tool = osim.InverseDynamicsTool()
    tool.setModelFileName(model_file)
    tool.setStartTime(start_time)
    tool.setEndTime(end_time)
    tool.setCoordinatesFileName(ik_path)
    tool.setExternalLoadsFileName(xml_path)
    tool.setResultsDir(os.path.dirname(output_mot_path))
    tool.setOutputGenForceFileName(os.path.basename(output_mot_path))
    return tool


def plot_id_moment(id_mot_path: str, out_png: str, side: str):
    mot = MOT.load_from_mot(id_mot_path)
    df = mot.data
    if "time" not in df.columns:
        return
    t = df["time"].to_numpy()
    col = "ankle_angle_r_moment" if side == "Right" else "ankle_angle_l_moment"
    plt.figure(figsize=(10, 4))
    if col in df.columns:
        plt.plot(t, df[col], label=col)
    plt.xlabel("Time (s)")
    plt.ylabel("Moment (N·m)")
    plt.title(os.path.basename(id_mot_path))
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
    INFO_CSV_NAME = "Trials_PLB_03.csv"

    participant_root = os.path.join(DATA_ROOT, PARTICIPANT)
    model_file = os.path.join(participant_root, "models", SCALED_MODEL_NAME)

    processed_root = os.path.join(participant_root, "processed")
    segmented_root = os.path.join(processed_root, "segmented")
    ik_root = os.path.join(processed_root, "ik")
    id_root = os.path.join(processed_root, "id")
    exloads_root = os.path.join(processed_root, "external_loads")
    plots_root = os.path.join(processed_root, "plots", "id")
    manifest_path = os.path.join(processed_root, "manifests", "overground_cycles_manifest.csv")

    infosheet_path = os.path.join(participant_root, "infosheet", INFO_CSV_NAME)
    if not os.path.exists(infosheet_path):
        raise FileNotFoundError(f"Infosheet not found: {infosheet_path}")

    infosheet = OvergroundInfoSheet(infosheet_path)

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Scaled model not found: {model_file}")

    dfm = read_manifest(manifest_path)
    if dfm is not None:
        iterator = iter_cycles_from_manifest(dfm)
        print(f"[ID] Using manifest: {manifest_path}")
    else:
        iterator = iter_segmented_grf(segmented_root)
        print(f"[ID] Using directory scan: {segmented_root}")

    n_ok = 0
    n_fail = 0

    for trial, side, plate_num, grf_path, trc_path in iterator:
        cycle_name = os.path.splitext(os.path.basename(grf_path))[0]

        # IK path (prefer raw)
        ik_path = os.path.join(ik_root, trial, side, f"FP{plate_num}", f"{cycle_name}_ik_raw.mot")
        if not os.path.exists(ik_path):
            ik_path = os.path.join(ik_root, trial, side, f"FP{plate_num}", f"{cycle_name}.mot")
        if not os.path.exists(ik_path):
            print(f"[ID] Missing IK: {ik_path}")
            n_fail += 1
            continue

        # time window: TRC master, clamp to GRF overlap
        try:
            trc_t0, trc_t1 = get_time_range_from_trc(trc_path)
            grf_t0, grf_t1 = get_time_range_from_grf(grf_path)
        except Exception as e:
            print(f"[ID] Skip: cannot read time {trial}/{side}/FP{plate_num} -> {repr(e)}")
            n_fail += 1
            continue

        start_time = max(trc_t0, grf_t0)
        end_time = min(trc_t1, grf_t1)
        if end_time <= start_time:
            print(f"[ID] Skip: no overlap window for {trial}/{side}/FP{plate_num} ({start_time:.4f}-{end_time:.4f})")
            n_fail += 1
            continue

        xml_path = os.path.join(exloads_root, trial, side, f"FP{plate_num}", f"{cycle_name}.xml")
        xml_ok = build_external_loads_xml_single_plate(
            grf_mot_path=grf_path,
            xml_path=xml_path,
            trial=trial,
            side=side,
            plate_num=plate_num,
            infosheet=infosheet,
            min_peak_vy_n=20.0
        )
        if xml_ok is None:
            continue

        out_dir = os.path.join(id_root, trial, side, f"FP{plate_num}")
        safe_mkdir(out_dir)
        output_mot_path = os.path.join(out_dir, f"{cycle_name}_id.mot")

        print(f"[ID] Running: {trial}/{side}/FP{plate_num}/{cycle_name}")
        tool = setup_id_tool(model_file, start_time, end_time, ik_path, xml_ok, output_mot_path)

        try:
            tool.run()
        except Exception as e:
            print(f"[ID] FAILED: {trial}/{side}/FP{plate_num}/{cycle_name} -> {repr(e)}")
            n_fail += 1
            continue

        if os.path.exists(output_mot_path):
            n_ok += 1
            out_png = os.path.join(plots_root, trial, side, f"FP{plate_num}", f"{cycle_name}_id.png")
            plot_id_moment(output_mot_path, out_png, side)
            print(f"[ID] Saved: {output_mot_path}")
        else:
            print(f"[ID] WARNING: OpenSim ran but output not found: {output_mot_path}")
            n_fail += 1

    print(f"\n[Done] ID completed. OK={n_ok}, FAIL={n_fail}")


if __name__ == "__main__":
    main()
