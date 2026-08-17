import os
import sys
from TreadMetrix import joint_power_computing
from resources.trial_class import Trial, GaitCycle


def recompute_power():
    trial_name = "k6 speed test"
    output_path = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test"
    
    # We need a trial object populated with ik and id cycles.
    # The easiest way is to use `Trial.to_gait_cycles` to load all the files.
    trial = Trial(mot=os.path.join(output_path, "corrected_mot", "k6 speed test.mot"), name=trial_name)
    
    for side in ["Right", "Left"]:
        ik_dir = os.path.join(output_path, "ik_results", side)
        id_dir = os.path.join(output_path, "id_results", side)
        
        # This will correctly populate the cycles using Trial's internal loaders
        cycles = GaitCycle.to_gait_cycles(side, ik_path=ik_dir, id_path=id_dir)
        
        if side == "Right":
            trial.add_cycles(cycles, [])
        else:
            trial.add_cycles([], cycles)
            
    power_output_path = os.path.join(output_path, "power_filtered_corrected")
    os.makedirs(power_output_path, exist_ok=True)
    
    print("Recomputing joint powers...")
    joint_power_computing.process(trial, power_output_path)
    print("Done!")

if __name__ == "__main__":
    recompute_power()
