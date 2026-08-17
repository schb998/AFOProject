import os
import sys
import re

# Ensure TreadMetrix is in path
sys.path.insert(0, r"D:\AFO_Codes")
sys.path.insert(0, r"D:\AFO_Codes\TreadMetrix")

from resources.file_types.mot import MOT
from TreadMetrix.id_computing import compute_external_loads, setup_id_tool
import opensim as osim

def run_id():
    ik_dir = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\ik_results"
    mot_dir = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\mot_corrected"
    model_file = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\P03-modle-scaled\Finalscaled.osim"
    
    base_dir = r"Z:\AFO\Collected Data\P03-Processed\P03\P03"
    id_results_dir = os.path.join(base_dir, "id_results")
    ext_loads_dir = os.path.join(base_dir, "external_loads")
    
    # We create the directories
    os.makedirs(id_results_dir, exist_ok=True)
    os.makedirs(ext_loads_dir, exist_ok=True)

    for side in ["Right", "Left"]:
        ik_side_dir = os.path.join(ik_dir, side)
        mot_side_dir = os.path.join(mot_dir, side)
        
        if not os.path.exists(ik_side_dir):
            continue
            
        out_id_side = os.path.join(id_results_dir, side)
        out_ext_side = os.path.join(ext_loads_dir, side)
        
        os.makedirs(out_id_side, exist_ok=True)
        os.makedirs(out_ext_side, exist_ok=True)
        
        ik_files = [f for f in os.listdir(ik_side_dir) if f.endswith(".mot")]
        for ik_file in ik_files:
            m = re.search(r"cycle(\d+)\.mot", ik_file)
            if not m:
                continue
            cycle_num = m.group(1)
            
            mot_cycle_num = int(cycle_num) + 1
            mot_file = f"afo speed 0_cycle{mot_cycle_num}.mot"
            mot_path = os.path.join(mot_side_dir, mot_file)
            ik_path = os.path.join(ik_side_dir, ik_file)
            
            if not os.path.exists(mot_path):
                print(f"MOT file missing for {ik_file}: {mot_path}")
                continue
            
            print(f"Processing side {side}, cycle {cycle_num}...")
            
            # Load MOT to get df and times
            try:
                mot_obj = MOT.load_from_mot(mot_path)
                df = mot_obj.data
            except Exception as e:
                print(f"Error loading MOT {mot_path}: {e}")
                continue
            
            time_col = 'time' if 'time' in df.columns else df.columns[0]
            start_time = float(df[time_col].iloc[0])
            end_time = float(df[time_col].iloc[-1])
            
            # Name of trial for XML
            trial_name = "P03_Gait01"
            xml_file_path = os.path.join(out_ext_side, f"{trial_name}_{side}_cycle{cycle_num}.xml")
            
            # Compute external loads
            try:
                compute_external_loads(df, mot_path, xml_file_path)
            except Exception as e:
                print(f"Error computing external loads for cycle {cycle_num}: {e}")
                continue
            
            output_mot = f"{trial_name}_{side}_cycle{cycle_num}.mot"
            id_tool = setup_id_tool(model_file, start_time, end_time, ik_path, xml_file_path, out_id_side, output_mot)
            
            try:
                id_tool.run()
            except Exception as e:
                print(f"Error running ID for cycle {cycle_num}: {e}")

if __name__ == '__main__':
    run_id()
