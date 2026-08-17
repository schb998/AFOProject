import os
import pathlib
import numpy as np
from scipy.signal import butter, filtfilt
import opensim as osim
from matplotlib import pyplot as plt
from resources.trial_class import Trial
from resources.file_types.mot import MOT
import resources.paths.paths_access as c

import numpy as np
from scipy.signal import butter, filtfilt

def ensure_grf_filtered_6hz(grf_obj, cutoff=6.0, order=4):
    """Applies a 6.0 Hz low-pass zero-phase Butterworth filter on GRF force signals prior to ID."""
    if grf_obj is None or grf_obj.data is None or len(grf_obj.data) <= 3:
        return
    time_col = 'time' if 'time' in grf_obj.data.columns else grf_obj.data.columns[0]
    fs = 1.0 / float(np.mean(np.diff(grf_obj.data[time_col].values)))
    nyq = 0.5 * fs
    normal_cutoff = min(0.99, cutoff / nyq)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    padlen = min(15, len(grf_obj.data) - 1)
    df = grf_obj.data.copy()
    for col in df.columns:
        if col.lower() == 'time':
            continue
        df[col] = filtfilt(b, a, df[col], padlen=padlen)
    grf_obj.data = df

def compute_external_loads(df, grf_path, xml_file_path, side: str = None):
    external_loads = osim.ExternalLoads()
    external_loads.setDataFileName(grf_path)

    side_lower = side.lower() if side else ""
    is_left = "left" in side_lower or side_lower == "l"
    is_right = "right" in side_lower or side_lower == "r"

    # left side (FP4 -> calcn_l)
    if (is_left or (not is_right and df[['ground_force4_vx', 'ground_force4_vy', 'ground_force4_vz']].abs().sum().sum() > 0)):
        ext1 = osim.ExternalForce()
        ext1.setName("FP4")
        ext1.set_applied_to_body("calcn_l")
        ext1.set_force_expressed_in_body("ground")
        ext1.set_point_expressed_in_body("ground")
        ext1.set_force_identifier("ground_force4_v")
        ext1.set_point_identifier("ground_force4_p")
        ext1.set_torque_identifier("ground_torque4_")
        ext1.set_data_source_name(grf_path)
        external_loads.cloneAndAppend(ext1)

    # right side (FP5 -> calcn_r)
    if (is_right or (not is_left and df[['ground_force5_vx', 'ground_force5_vy', 'ground_force5_vz']].abs().sum().sum() > 0)):
        ext2 = osim.ExternalForce()
        ext2.setName("FP5")
        ext2.set_applied_to_body("calcn_r")
        ext2.set_force_expressed_in_body("ground")
        ext2.set_point_expressed_in_body("ground")
        ext2.set_force_identifier("ground_force5_v")
        ext2.set_point_identifier("ground_force5_p")
        ext2.set_torque_identifier("ground_torque5_")
        ext2.set_data_source_name(grf_path)
        external_loads.cloneAndAppend(ext2)

    # save the external loads
    print(f"Created: {xml_file_path}")
    external_loads.printToXML(xml_file_path)
    return external_loads

def setup_id_tool(scaled_model_file, start_time, end_time, ik_path, xml_file_path, output_directory, output_file):
    id_tool = osim.InverseDynamicsTool()
    id_tool.setModelFileName(scaled_model_file)
    id_tool.setStartTime(start_time)
    id_tool.setEndTime(end_time)
    id_tool.setCoordinatesFileName(ik_path)
    id_tool.setExternalLoadsFileName(xml_file_path)
    id_tool.setResultsDir(output_directory)
    id_tool.setOutputGenForceFileName(output_file)
    return id_tool


def process(trial: Trial,
            external_loads_path: str,
            id_results_path: str,
            scaled_model_file: str) -> None:

    name = trial.name
    for side in ["Right", "Left"]:
        cycles = trial.gait_cycles[side]
        temp_path = os.path.join(id_results_path, side, "temp")
        side_xml = os.path.join(external_loads_path, side)
        side_out = os.path.join(id_results_path, side)
        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(side_xml, exist_ok=True)
        os.makedirs(side_out, exist_ok=True)

        for cycle in cycles:
            error_message = f"Couldn't compute Internal Dynamics of the trial {name} for the gait cycle {side}, {cycle.num}:"

            grf = cycle.grf
            trc = cycle.trc
            ik = cycle.ik

            if grf is None or ik is None:
                print(error_message + "insufficient data in trial.")
                break

            # Read start/end time from IK
            start_time = float(ik.data['time'].iloc[0])
            end_time = float(ik.data['time'].iloc[-1])

            # Apply 6 Hz low-pass filter to GRF forces before ID calculation
            ensure_grf_filtered_6hz(grf, cutoff=6.0, order=4)
            df = grf.data

            if grf.filepath is None or not os.path.exists(grf.filepath):
                grf.save(temp_path)
            else:
                # Re-save to guarantee file on disk contains 6 Hz filtered GRF forces
                grf.save(os.path.dirname(grf.filepath))
            grf_path = grf.filepath

            xml_file_path = os.path.join(side_xml, f"{name}_{side}_cycle{cycle.num}.xml")
            external_loads = compute_external_loads(df, grf_path, xml_file_path, side=side)
            cycle.add_external_loads(external_loads_path=xml_file_path, exl_object=external_loads)

            if ik.filepath is None or not os.path.exists(ik.filepath):
                ik.save(temp_path)
            ik_path = ik.filepath

            # Run Inverse Dynamics
            print(f"Running ID: {name}/{side}/{cycle.num}")
            output_mot = f"{name}_{side}_cycle{cycle.num}.mot"
            id_tool = setup_id_tool(scaled_model_file, start_time, end_time, ik_path, xml_file_path, side_out, output_mot)
            try:
                id_tool.run()
                print(f"Saved: {output_mot}")
                id_full_path = os.path.join(side_out, output_mot)

                # Subject weight normalization for ID moments if participant weight is specified
                subject_weight = c.get_subject_weight()
                if subject_weight is not None and subject_weight > 0 and os.path.exists(id_full_path):
                    try:
                        id_mot = MOT.load_from_mot(id_full_path)
                        for col in id_mot.data.columns:
                            if col.lower() != 'time' and '_moment' in col.lower():
                                id_mot.data[col] = id_mot.data[col] / subject_weight
                        id_mot.save(side_out)
                        cycle.add_id(id_full_path, id_mot)
                        print(f"  [Normalized] ID moments by participant weight ({subject_weight} kg) -> {output_mot}")
                    except Exception as norm_err:
                        print(f"  Warning: Could not weight-normalize ID file {output_mot}: {norm_err}")
                        cycle.add_id(id_full_path)
                else:
                    cycle.add_id(id_full_path)

            except Exception as e:
                print(f"Error for {name}: {e}")

            for file in os.listdir(temp_path):
                os.remove(os.path.join(temp_path, file))


        try:
            pathlib.Path.rmdir(pathlib.Path(temp_path))
        except OSError:
            print(f"Error deleting temporary directory {temp_path}.")

    print(f"Internal Dynamics for trial {name} processed.")

