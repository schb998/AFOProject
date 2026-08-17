import os
import sys

# Ensure TreadMetrix is in path
sys.path.insert(0, r"D:\AFO_Codes")
sys.path.insert(0, r"D:\AFO_Codes\TreadMetrix")

from resources.trial_class import Trial, GaitCycle
from data_postprocessing import process as post_processing
from id_computing import process as compute_id
from joint_power_computing import process as compute_jp

def run_id_and_power():
    # 1. Setup paths
    raw_dir = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\Gait01_rotated"
    mot_file = os.path.join(raw_dir, "afo speed 0.mot")
    trc_file = os.path.join(raw_dir, "afo speed 0.trc")
    
    scaled_model_file = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\P03-modle-scaled\Finalscaled.osim"
    
    # Set output directories matching the original structure
    base_dir = r"Z:\AFO\Collected Data\P03-Processed\P03\P03"
    corrected_mot_path = os.path.join(base_dir, "mot_corrected")
    segmented_path = base_dir 
    
    ik_results_path = os.path.join(base_dir, "ik_results")
    id_results_path = os.path.join(base_dir, "id_results")
    ext_loads_path = os.path.join(base_dir, "external_loads")
    power_filtered_path = os.path.join(base_dir, "power_filtered")
    
    os.makedirs(id_results_path, exist_ok=True)
    os.makedirs(ext_loads_path, exist_ok=True)
    os.makedirs(power_filtered_path, exist_ok=True)

    # 2. Load Trial with the rotated data
    print("Loading Trial with rotated data...")
    trial_name = "P03_Gait01"
    trial = Trial(mot=mot_file)
    trial.name = trial_name
    trial.add_trc(trc_file)
    
    # We must quickly re-run segmentation just to populate trial.gait_cycles correctly
    # but we will NOT run IK. Segmentation is fast.
    print("Populating Gait Cycles (Skipping IK)...")
    post_processing(trial, save_plot_path=corrected_mot_path,
                    save_segmented_path=segmented_path,
                    show=False, save_optionals=False)
    
    # 3. Match IK files to cycles
    print("Setting up Trial object for ID matching...")
    for side in ["Right", "Left"]:
        if side in trial.gait_cycles:
            cycles = trial.gait_cycles[side]
            side_ik_dir = os.path.join(ik_results_path, side)
            
            if os.path.exists(side_ik_dir):
                ik_files = [f for f in os.listdir(side_ik_dir) if f.endswith(".mot")]
                for cycle in cycles:
                    expected_ik_name = f"{trial.name}_{side.lower()}_cycle{cycle.num}.mot"
                    if expected_ik_name in ik_files:
                        # IMPORTANT FIX: Use add_ik instead of assigning a string!
                        cycle.add_ik(inverse_kinematic_path=os.path.join(side_ik_dir, expected_ik_name))
                    else:
                        print(f"Warning: Could not find IK file {expected_ik_name}")
    
    # 4. Compute ID
    print("Running ID Computing...")
    compute_id(trial, ext_loads_path, id_results_path, scaled_model_file)
    
    # 5. Compute Joint Power
    print("Running Joint Power Computing...")
    compute_jp(trial, power_filtered_path)
    
    print("ID and Joint Power computations completed successfully!")

if __name__ == '__main__':
    run_id_and_power()
