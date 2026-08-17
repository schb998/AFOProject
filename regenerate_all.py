import os
from TreadMetrix import data_postprocessing, id_computing, joint_power_computing
from resources.trial_class import Trial, GaitCycle
from resources.file_types.mot import MOT
import shutil

def main():
    trial_name = "k6 speed test"
    output_path = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test"
    
    # We already have the completely unfiltered/uncorrected MOT? No, we have corrected_mot.
    # But corrected_mot was already zeroed out incorrectly!
    # We must use the original MOT to start over.
    raw_mot_path = r"Z:\AFO\Collected Data\P03\P03\Gait01\k5 speed test.mot"
    trial = Trial(mot=raw_mot_path, name=trial_name)
    
    frame_rate = 1000.0 # Standard GRF frame rate
    
    # 1. Post-process (Filter & Baseline correct)
    corrected_grf = trial.grf.copy()
    corrected_grf.rename(name=trial.name, filename=trial.name + ".mot")
    data_postprocessing.filter_grf(corrected_grf, frame_rate)
    
    # The actual baseline correction in data_postprocessing does exactly this:
    data_postprocessing.baseline_correct_debug(corrected_grf, 'ground_force2_vy', ['ground_force2_vx', 'ground_force2_vz'], show=False)
    data_postprocessing.baseline_correct_debug(corrected_grf, 'ground_force1_vy', ['ground_force1_vx', 'ground_force1_vz'], show=False)
    
    # Detect
    toe_off_moments = data_postprocessing.detect_toe_offs(corrected_grf, frame_rate)
    heel_strike_moments = data_postprocessing.detect_heel_strikes(corrected_grf, frame_rate)
    
    # Zero swing phase
    data_postprocessing.zero_swing_phase(corrected_grf, toe_off_moments, heel_strike_moments, 'right')
    data_postprocessing.zero_swing_phase(corrected_grf, toe_off_moments, heel_strike_moments, 'left')
    
    # Re-Segment
    start_idx = 29701
    end_idx = 68416
    corrected_grf = corrected_grf.sample(start_idx, end_idx)
    for side in ["L", "R"]:
        heel_strike_moments[side] = [strike for strike in heel_strike_moments[side]
                                     if (start_idx <= strike <= end_idx)]
                                     
    right_mots = corrected_grf.segment(heel_strike_moments['R'], True)[1:-1]
    left_mots = corrected_grf.segment(heel_strike_moments['L'], True)[1:-1]
    
    right_path = os.path.join(output_path, "segmented", "Right")
    left_path = os.path.join(output_path, "segmented", "Left")
    
    if os.path.exists(right_path): shutil.rmtree(right_path)
    if os.path.exists(left_path): shutil.rmtree(left_path)
    os.makedirs(right_path, exist_ok=True)
    os.makedirs(left_path, exist_ok=True)
    
    MOT.save_multiple(right_mots, right_path)
    MOT.save_multiple(left_mots, left_path)
    
    trial.gait_cycles["Right"] = GaitCycle.to_gait_cycles(side="Right", grfs=right_mots, grf_path=right_path)
    trial.gait_cycles["Left"] = GaitCycle.to_gait_cycles(side="Left", grfs=left_mots, grf_path=left_path)
    
    # Link the IK data which is already processed and correct!
    ik_dir_right = os.path.join(output_path, "ik_results", "Right")
    ik_dir_left = os.path.join(output_path, "ik_results", "Left")
    
    ik_cycles_right = GaitCycle.to_gait_cycles("Right", ik_path=ik_dir_right)
    ik_cycles_left = GaitCycle.to_gait_cycles("Left", ik_path=ik_dir_left)
    
    for i in range(min(len(trial.gait_cycles["Right"]), len(ik_cycles_right))):
        trial.gait_cycles["Right"][i].ik = ik_cycles_right[i].ik
        
    for i in range(min(len(trial.gait_cycles["Left"]), len(ik_cycles_left))):
        trial.gait_cycles["Left"][i].ik = ik_cycles_left[i].ik
    
    # 2. Run ID
    print("Running ID...")
    scaled_model = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\P03-modle-scaled\K6_scaled.osim"
    external_loads_path = os.path.join(output_path, "external_loads")
    id_results_path = os.path.join(output_path, "id_results")
    if os.path.exists(id_results_path): shutil.rmtree(id_results_path)
    os.makedirs(id_results_path, exist_ok=True)
    
    id_computing.process(trial, external_loads_path, id_results_path, scaled_model)
    
    # 3. Run Joint Power
    print("Running Joint Power...")
    power_output_path = os.path.join(output_path, "power_filtered_corrected")
    if os.path.exists(power_output_path): shutil.rmtree(power_output_path)
    os.makedirs(power_output_path, exist_ok=True)
    joint_power_computing.process(trial, power_output_path)
    print("DONE!")

if __name__ == "__main__":
    main()
