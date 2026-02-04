import os
import pathlib
import numpy as np
import pandas as pd
import opensim as osim

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_segmented_cycles(segmented_root: str):
    """
    Expects:
      segmented/<Trial>/<Side>/<FPx>/*.trc
      segmented/<Trial>/<Side>/<FPx>/*.mot   (GRF)

    Yields:
      (trial_name, side, fp_folder, trc_path, grf_path)
    """
    for trial_name in os.listdir(segmented_root):
        tdir = os.path.join(segmented_root, trial_name)
        if not os.path.isdir(tdir):
            continue

        for side in ("Right", "Left"):
            sdir = os.path.join(tdir, side)
            if not os.path.isdir(sdir):
                continue

            for fp_folder in os.listdir(sdir):
                fpdir = os.path.join(sdir, fp_folder)
                if not os.path.isdir(fpdir):
                    continue

                # pair trc + grf mot by basename
                trcs = {os.path.splitext(f)[0]: os.path.join(fpdir, f)
                        for f in os.listdir(fpdir) if f.lower().endswith(".trc")}
                mots = {os.path.splitext(f)[0]: os.path.join(fpdir, f)
                        for f in os.listdir(fpdir) if f.lower().endswith(".mot")}

                # Only yield pairs that exist in both
                for base, trc_path in trcs.items():
                    if base in mots:
                        yield trial_name, side, fp_folder, trc_path, mots[base]


def corresponding_ik_path(ik_root: str, trial_name: str, side: str, fp_folder: str, base: str) -> str:
    """
    IK layout:
      ik/<Trial>/<Side>/<FPx>/<base>.mot
    """
    return os.path.join(ik_root, trial_name, side, fp_folder, f"{base}.mot")


#  GRF plate detection

def detect_active_grf_plate(grf_df: pd.DataFrame, min_peak_N: float = 20.0) -> int | None:
    """
    Picks the plate with the highest peak vertical GRF.
    Returns 1/2/3 or None.
    """
    best_plate = None
    best_peak = -1.0

    for plate in (1, 2, 3):
        vy = f"ground_force{plate}_vy"
        if vy not in grf_df.columns:
            continue
        peak = float(grf_df[vy].max())
        if peak > best_peak:
            best_peak = peak
            best_plate = plate

    if best_peak < min_peak_N:
        return None
    return best_plate


# ---------- ExternalLoads XML creation ----------

def write_external_loads_xml(
    grf_df: pd.DataFrame,
    grf_mot_path: str,
    xml_out_path: str,
    side: str,
    min_peak_N: float = 20.0
) -> int | None:
    """
    Creates ExternalLoads with exactly ONE ExternalForce:
      - the plate with largest peak vertical force.
    Returns the selected plate (1/2/3) or None if no usable GRF.
    """

    plate = detect_active_grf_plate(grf_df, min_peak_N=min_peak_N)
    if plate is None:
        return None

    applied_body = "calcn_r" if side.lower().startswith("r") else "calcn_l"

    ext_loads = osim.ExternalLoads()
    ext_loads.setDataFileName(grf_mot_path)

    ext = osim.ExternalForce()
    ext.setName(f"FP{plate}")
    ext.set_applied_to_body(applied_body)
    ext.set_force_expressed_in_body("ground")
    ext.set_point_expressed_in_body("ground")

    # IMPORTANT: matches your GRF headers
    ext.set_force_identifier(f"ground_force{plate}_v")
    ext.set_point_identifier(f"ground_force{plate}_p")
    ext.set_torque_identifier(f"ground_torque{plate}_")

    ext_loads.cloneAndAppend(ext)

    safe_mkdir(os.path.dirname(xml_out_path))
    ext_loads.printToXML(xml_out_path)

    return plate


#ID tool setup ----------

def setup_id_tool(
    scaled_model_file: str,
    start_time: float,
    end_time: float,
    ik_path: str,
    external_loads_xml: str,
    results_dir: str,
    output_file: str
) -> osim.InverseDynamicsTool:

    id_tool = osim.InverseDynamicsTool()
    id_tool.setModelFileName(scaled_model_file)
    id_tool.setStartTime(start_time)
    id_tool.setEndTime(end_time)
    id_tool.setCoordinatesFileName(ik_path)
    id_tool.setExternalLoadsFileName(external_loads_xml)
    id_tool.setResultsDir(results_dir)
    id_tool.setOutputGenForceFileName(output_file)
    return id_tool


# ---------- MAIN ID pipeline ----------

def main():
    DATA_ROOT = r"D:\TestOverground\Overground"
    PARTICIPANT = "PLB_03"
    SCALED_MODEL_NAME = "scaledmodelIM.osim"

    participant_root = os.path.join(DATA_ROOT, PARTICIPANT)
    scaled_model_file = os.path.join(participant_root, "models", SCALED_MODEL_NAME)

    processed_root = os.path.join(participant_root, "processed")
    segmented_root = os.path.join(processed_root, "segmented")
    ik_root = os.path.join(processed_root, "ik")

    id_root = os.path.join(processed_root, "id")
    exloads_root = os.path.join(processed_root, "external_loads")

    if not os.path.exists(scaled_model_file):
        raise FileNotFoundError(f"Scaled model not found: {scaled_model_file}")

    n_done = 0
    n_skip = 0

    for trial_name, side, fp_folder, trc_path, grf_path in iter_segmented_cycles(segmented_root):
        base = os.path.splitext(os.path.basename(trc_path))[0]  # same as mot basename

        ik_path = corresponding_ik_path(ik_root, trial_name, side, fp_folder, base)
        if not os.path.exists(ik_path):
            print(f"[SKIP] Missing IK: {ik_path}")
            n_skip += 1
            continue

        # Load TRC to get time range
        trc = TRC.load_from_trc(trc_path)
        start_time = float(trc.data["Time"].iloc[0])
        end_time = float(trc.data["Time"].iloc[-1])

        # Load GRF df for XML creation
        grf = MOT.load_from_mot(grf_path)
        grf_df = grf.data

        xml_path = os.path.join(exloads_root, trial_name, side, fp_folder, f"{base}.xml")
        used_plate = write_external_loads_xml(
            grf_df=grf_df,
            grf_mot_path=grf_path,
            xml_out_path=xml_path,
            side=side,
            min_peak_N=20.0
        )

        if used_plate is None:
            print(f"[SKIP] No usable GRF plate detected for: {trial_name}/{side}/{fp_folder}/{base}")
            n_skip += 1
            continue

        out_dir = os.path.join(id_root, trial_name, side, fp_folder)
        safe_mkdir(out_dir)

        out_mot = f"{base}_id.mot"
        print(f"[ID] {trial_name}/{side}/{fp_folder}  plate=FP{used_plate}  -> {out_mot}")

        id_tool = setup_id_tool(
            scaled_model_file=scaled_model_file,
            start_time=start_time,
            end_time=end_time,
            ik_path=ik_path,
            external_loads_xml=xml_path,
            results_dir=out_dir,
            output_file=out_mot
        )

        try:
            id_tool.run()
            n_done += 1
        except Exception as e:
            print(f"[FAIL] ID failed for {trial_name}/{side}/{fp_folder}/{base}: {repr(e)}")
            n_skip += 1

    print("\n[Done] ID completed.")
    print(f"[Done] Successful: {n_done}")
    print(f"[Done] Skipped/failed: {n_skip}")


if __name__ == "__main__":
    main()