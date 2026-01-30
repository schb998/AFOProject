import os
import pathlib
import opensim as osim
from matplotlib import pyplot as plt
from resources.trial_class import Trial

# todo: compute external loads method parameters df & grf_path can be switched for a single MOT object


def compute_external_loads(df, grf_path, xml_file_path) -> osim.ExternalLoads:
    """Calls on OpenSim to compute the external loads

    Args:
        df: pd.dataFrame, grf data
        grf_path: str, path to the GRF  file
        xml_file_path: str, output path to saev the external loads

    Returns:
        osim external loads object

    """
    external_loads = osim.ExternalLoads()
    external_loads.setDataFileName(grf_path)
    # left side
    if df[['ground_force1_vx', 'ground_force1_vy', 'ground_force1_vz']].abs().sum().sum() > 0:
        ext1 = osim.ExternalForce()
        ext1.setName("FP1")
        ext1.set_applied_to_body("calcn_l")
        ext1.set_force_expressed_in_body("ground")
        ext1.set_point_expressed_in_body("ground")
        ext1.set_force_identifier("ground_force1_v")
        ext1.set_point_identifier("ground_force1_p")
        ext1.set_torque_identifier("ground_torque1_")
        external_loads.cloneAndAppend(ext1)

    # right side
    if df[['ground_force2_vx', 'ground_force2_vy', 'ground_force2_vz']].abs().sum().sum() > 0:
        ext2 = osim.ExternalForce()
        ext2.setName("FP2")
        ext2.set_applied_to_body("calcn_r")
        ext2.set_force_expressed_in_body("ground")
        ext2.set_point_expressed_in_body("ground")
        ext2.set_force_identifier("ground_force2_v")
        ext2.set_point_identifier("ground_force2_p")
        ext2.set_torque_identifier("ground_torque2_")
        external_loads.cloneAndAppend(ext2)

    # save the external loads
    print(f"Created: {xml_file_path}")
    external_loads.printToXML(xml_file_path)
    return external_loads

def setup_id_tool(scaled_model_file, start_time, end_time, ik_path, xml_file_path, output_directory, output_file) \
        -> osim.InverseDynamicsTool:
    """Sets up OpenSim's Ik tool"""
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
    """compute the id of the given trial"""

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

            if grf is None or trc is None or ik is None:
                print(error_message + "insufficient data in trial.")
                break

            # Read start/end time from TRC
            start_time = float(trc.data['Time'][trc.first_frame])
            end_time = float(trc.data['Time'][trc.first_frame + trc.data.shape[0] - 1])

            # Generate ExternalLoads XML
            df = grf.data
            if grf.filepath is None:
                grf.save(temp_path)
            grf_path = grf.filepath

            xml_file_path = os.path.join(side_xml, f"{name}_exloads_{side}_{cycle.num}.xml")
            external_loads = compute_external_loads(df, grf_path, xml_file_path)
            cycle.add_external_loads(external_loads_path=xml_file_path, exl_object=external_loads)

            if ik.filepath is None:
                ik.save(temp_path)
            ik_path = ik.filepath

            # Run Inverse Dynamics
            print(f"Running ID: {name}/{side}/{cycle.num}")
            output_mot = f"{name}_ID_{side}_{cycle.num}.mot"
            id_tool = setup_id_tool(scaled_model_file, start_time, end_time, ik_path, xml_file_path, side_out, output_mot)

            try:
                id_tool.run()
                print(f"Saved: {output_mot}")
                cycle.add_id(os.path.join(side_out, output_mot))

                # plt.plot(cycle.id.data['time'], cycle.id.data['ankle_angle_r_moment'])
                # plt.show()

            except Exception as e:
                print(f"Error for {name}: {e}")

            for file in os.listdir(temp_path):
                os.remove(os.path.join(temp_path, file))


        try:
            pathlib.Path.rmdir(pathlib.Path(temp_path))
        except OSError:
            print(f"Error deleting temporary directory {temp_path}.")

    print(f"Internal Dynamics for trial {name} processed.")

