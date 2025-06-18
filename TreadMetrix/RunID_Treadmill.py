import os
import numpy as np
import pandas as pd
import opensim as osim
from read_trc import read_trc

# OpenSim
opensim_path = r"C:/OpenSim 4.4/bin"
os.environ['OPENSIM_HOME'] = opensim_path
os.add_dll_directory(opensim_path)

base_path = r"D:\MyData\Testingworkflow\Test"
model_file = os.path.join(base_path, "scaled_model_Ella.osim")
trc_path = os.path.join(base_path, "segmented_trc")
ik_path = os.path.join(base_path, "IK_Results")
grf_path = os.path.join(base_path, "mot_chopped")
external_loads_path = os.path.join(base_path, "external_loads")
id_results_path = os.path.join(base_path, "ID_Results")

for side in ["Right", "Left"]:
    os.makedirs(os.path.join(external_loads_path, side), exist_ok=True)
    os.makedirs(os.path.join(id_results_path, side), exist_ok=True)


for side in ["Right", "Left"]:
    side_trc = os.path.join(trc_path, side)
    side_ik = os.path.join(ik_path, side)
    side_grf = os.path.join(grf_path, side)
    side_xml = os.path.join(external_loads_path, side)
    side_out = os.path.join(id_results_path, side)

    for trc_file in sorted(os.listdir(side_trc)):
        if not trc_file.endswith(".trc"):
            continue

        name = trc_file.replace(".trc", "")
        trc_file_path = os.path.join(side_trc, trc_file)
        ik_file_path = os.path.join(side_ik, f"{name}.mot")
        grf_file_path = os.path.join(side_grf, f"{name}.mot")
        xml_file_path = os.path.join(side_xml, f"{name}.xml")
        output_mot = os.path.join(side_out, f"{name}.mot")

        if not os.path.exists(ik_file_path) or not os.path.exists(grf_file_path):
            print(f"Skipping {name} (missing IK or GRF)")
            continue

        # Read start/end time from TRC
        trc_data = read_trc(trc_file_path)
        start_time = float(trc_data[0]['Data']['Time'][0])
        end_time = float(trc_data[0]['Data']['Time'][-1])

        # Generate ExternalLoads XML
        df = pd.read_csv(grf_file_path, sep=r'\s+', skiprows=6)
        external_loads = osim.ExternalLoads()
        external_loads.setDataFileName(grf_file_path)

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

        external_loads.printToXML(xml_file_path)
        print(f"Created: {xml_file_path}")

        # Run Inverse Dynamics
        print(f"Running ID: {side}/{name}")
        id_tool = osim.InverseDynamicsTool()
        id_tool.setModelFileName(model_file)
        id_tool.setStartTime(start_time)
        id_tool.setEndTime(end_time)
        id_tool.setCoordinatesFileName(ik_file_path)
        id_tool.setExternalLoadsFileName(xml_file_path)
        id_tool.setResultsDir(side_out)
        id_tool.setOutputGenForceFileName(f"{name}.mot")

        try:
            id_tool.run()
            print(f"Saved: {output_mot}")
        except Exception as e:
            print(f"Error for {name}: {e}")
