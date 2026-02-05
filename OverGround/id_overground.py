import os
import pathlib
import opensim as osim
import pandas as pd
import sys

# Add project root to path to allow importing 'resources'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def find_first_csv(infosheet_dir: str) -> str:
    if not os.path.isdir(infosheet_dir):
        raise FileNotFoundError(f"Infosheet folder not found: {infosheet_dir}")

    csvs = [os.path.join(infosheet_dir, f) for f in os.listdir(infosheet_dir) if f.lower().endswith(".csv")]
    if len(csvs) == 0:
        raise FileNotFoundError(f"No .csv file found in infosheet folder: {infosheet_dir}")

    csvs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return csvs[0]


def parse_plate_num(fp_folder: str) -> int | None:
    s = fp_folder.strip().upper()
    if not s.startswith("FP"):
        return None
    try:
        return int(s.replace("FP", "").strip())
    except ValueError:
        return None


def find_same_stem_file(folder: str, stem: str, exts: tuple[str, ...]) -> str | None:
    if not os.path.isdir(folder):
        return None
    stem_low = stem.lower()
    exts_low = tuple(e.lower() for e in exts)
    for f in os.listdir(folder):
        base, ext = os.path.splitext(f)
        if base.lower() == stem_low and ext.lower() in exts_low:
            return os.path.join(folder, f)
    return None


def iter_segmented_cycles(segmented_root: str):
    """
    Expects:
      segmented/<Trial>/<Side>/<FPx>/
          <cycle_name>.mot   (GRF)
          <cycle_name>.trc   (TRC)
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
                    if not f.lower().endswith(".mot"):
                        continue
                    grf_mot_path = os.path.join(fpdir, f)
                    cycle_name = os.path.splitext(f)[0]

                    trc_path = find_same_stem_file(fpdir, cycle_name, exts=(".trc",))
                    if trc_path is None:
                        print(f"[SEG] Missing TRC for cycle: {grf_mot_path}")
                        continue

                    yield trial, side, fp, grf_mot_path, trc_path, cycle_name


def get_time_range_from_trc(trc_path: str) -> tuple[float, float]:
    trc = TRC.load_from_trc(trc_path)

    if trc.data is None or trc.data.shape[0] == 0:
        raise ValueError(f"TRC has no rows: {trc_path}")

    time_col = None
    for c in ["Time", "time"]:
        if c in trc.data.columns:
            time_col = c
            break
    if time_col is None:
        raise KeyError(f"No Time column found in TRC: {trc_path}. Columns: {list(trc.data.columns)}")

    start_time = float(trc.data[time_col].iloc[0])
    end_time = float(trc.data[time_col].iloc[-1])
    return start_time, end_time

# Infosheet

def load_infosheet_map(infosheet_csv: str) -> dict[str, dict]:
    df = pd.read_csv(infosheet_csv)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    trial_col = "Trials/Events"
    if trial_col not in df.columns:
        raise KeyError(f"Infosheet missing '{trial_col}' column. Found: {list(df.columns)}")

    sheet = {}
    for _, r in df.iterrows():
        trial_name = str(r[trial_col]).strip()
        row = {}
        for k, v in r.to_dict().items():
            if pd.isna(v):
                row[str(k).strip()] = ""
            else:
                row[str(k).strip()] = str(v).strip()
        sheet[trial_name] = row
    return sheet


def should_run_side(info_row: dict, side: str) -> bool:
    valid = info_row.get("Valid GaitCycle", "").strip().lower()
    if valid == "" or valid == "both":
        return True
    return valid == side.strip().lower()


def plate_flag_to_body(flag: str) -> str | None:
    f = (flag or "").strip().lower()
    if f == "left":
        return "calcn_l"
    if f == "right":
        return "calcn_r"
    if f == "both":
        return None
    return None


def plate_is_nonzero(df, plate_num: int, eps_sum: float = 1e-6) -> bool:
    cols = [f"ground_force{plate_num}_vx", f"ground_force{plate_num}_vy", f"ground_force{plate_num}_vz"]
    if any(c not in df.columns for c in cols):
        return False
    return float(df[cols].abs().sum().sum()) > eps_sum


def build_external_loads_from_infosheet(
    grf_mot_path: str,
    xml_path: str,
    info_row: dict,
    eps_sum: float = 1e-6,
    plate_nums: tuple[int, ...] = (1, 2, 3),
):
    mot = MOT.load_from_mot(grf_mot_path)
    df = mot.data

    ext_loads = osim.ExternalLoads()
    ext_loads.setDataFileName(grf_mot_path)

    added_any = False

    for p in plate_nums:
        flag = info_row.get(f"FP{p}", "")
        body = plate_flag_to_body(flag)

        if body is None:
            continue

        if not plate_is_nonzero(df, p, eps_sum=eps_sum):
            continue

        ext = osim.ExternalForce()
        ext.setName(f"FP{p}")
        ext.set_applied_to_body(body)
        ext.set_force_expressed_in_body("ground")
        ext.set_point_expressed_in_body("ground")
        ext.set_force_identifier(f"ground_force{p}_v")
        ext.set_point_identifier(f"ground_force{p}_p")
        ext.set_torque_identifier(f"ground_torque{p}_")

        ext_loads.cloneAndAppend(ext)
        added_any = True

    if not added_any:
        print(f"[EXLOADS] Skip: no valid non-zero plates (after infosheet) for {os.path.basename(grf_mot_path)}")
        return None

    safe_mkdir(os.path.dirname(xml_path))
    ext_loads.printToXML(xml_path)
    print(f"[EXLOADS] Wrote: {xml_path}")
    return xml_path

def setup_id_tool(
    model_file: str,
    start_time: float,
    end_time: float,
    ik_path: str,
    xml_path: str,
    output_mot_path: str,
) -> osim.InverseDynamicsTool:
    tool = osim.InverseDynamicsTool()
    tool.setModelFileName(model_file)
    tool.setStartTime(start_time)
    tool.setEndTime(end_time)
    tool.setCoordinatesFileName(ik_path)
    tool.setExternalLoadsFileName(xml_path)
    tool.setResultsDir(os.path.dirname(output_mot_path))
    tool.setOutputGenForceFileName(os.path.basename(output_mot_path))
    return tool


def main():
    print("[DEBUG] Starting main...")
    # Use relative path to resources/example which is at ../resources/example from this script
    DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "example"))
    PARTICIPANT = "PLB_03"
    SCALED_MODEL_NAME = "scaledmodelIM.osim"
    
    participant_root = os.path.join(DATA_ROOT, PARTICIPANT)
    INFOSHEET_DIR = os.path.join(participant_root, "infosheet")
    model_file = os.path.join(participant_root, "models", SCALED_MODEL_NAME)

    processed_root = os.path.join(participant_root, "processed")
    segmented_root = os.path.join(processed_root, "segmented")
    ik_root = os.path.join(processed_root, "ik")
    id_root = os.path.join(processed_root, "id")
    exloads_root = os.path.join(processed_root, "external_loads")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Scaled model not found: {model_file}")
    if not os.path.isdir(segmented_root):
        raise FileNotFoundError(f"Segmented folder not found: {segmented_root}")
    if not os.path.isdir(ik_root):
        raise FileNotFoundError(f"IK folder not found: {ik_root}")

    infosheet_csv = find_first_csv(INFOSHEET_DIR)
    print(f"[INFO] Using infosheet CSV: {infosheet_csv}")
    infosheet_map = load_infosheet_map(infosheet_csv)

    for trial, side, fp, grf_path, trc_path, cycle_name in iter_segmented_cycles(segmented_root):
        info = infosheet_map.get(trial)
        if info is None:
            print(f"[ID] Skip: no infosheet row for trial '{trial}'")
            continue

        if not should_run_side(info, side):
            print(f"[ID] Skip: trial {trial} side {side} (Valid GaitCycle={info.get('Valid GaitCycle','')})")
            continue

        ik_path = os.path.join(ik_root, trial, side, fp, f"{cycle_name}.mot")
        if not os.path.exists(ik_path):
            print(f"[ID] Missing IK: {ik_path}")
            continue

        # Master window from TRC
        try:
            start_time, end_time = get_time_range_from_trc(trc_path)
        except Exception as e:
            print(f"[ID] Skip: cannot read TRC time {trc_path} -> {repr(e)}")
            continue

        xml_path = os.path.join(exloads_root, trial, side, fp, f"{cycle_name}.xml")
        xml_ok = build_external_loads_from_infosheet(
            grf_mot_path=grf_path,
            xml_path=xml_path,
            info_row=info,
            eps_sum=1e-6,
            plate_nums=(1, 2, 3),
        )
        if xml_ok is None:
            continue

        out_dir = os.path.join(id_root, trial, side, fp)
        safe_mkdir(out_dir)
        output_mot_path = os.path.join(out_dir, f"{cycle_name}_id.mot")

        print(f"[ID] Running: {trial}/{side}/{fp}/{cycle_name}")
        print(f"     TRC window: {start_time:.6f} → {end_time:.6f}")
        print(f"     GRF file : {os.path.basename(grf_path)}")
        print(f"     TRC file : {os.path.basename(trc_path)}")
        print(f"     IK file  : {os.path.basename(ik_path)}")

        id_tool = setup_id_tool(model_file, start_time, end_time, ik_path, xml_ok, output_mot_path)

        try:
            id_tool.run()
        except Exception as e:
            print(f"[ID] FAILED: {trial}/{side}/{fp}/{cycle_name} -> {repr(e)}")
            continue

        if os.path.exists(output_mot_path):
            print(f"[ID] Saved: {output_mot_path}")
        else:
            print(f"[ID] WARNING: OpenSim ran but output not found: {output_mot_path}")

    print("\n[Done] ID completed.")


if __name__ == "__main__":
    main()
