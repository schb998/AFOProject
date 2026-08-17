import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevent GUI windows
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Setup OpenSim Environment
opensim_path = r"C:\OpenSim 4.4\bin"
os.environ['OPENSIM_HOME'] = opensim_path
os.add_dll_directory(opensim_path)
sys.path.append(os.path.join(r"C:\OpenSim 4.4", 'Bindings', 'Python'))
os.environ['PATH'] += os.pathsep + opensim_path

import opensim as osim

# Add project directories to path
sys.path.insert(0, r"D:\AFO_Codes")
sys.path.insert(0, r"D:\AFO_Codes\TreadMetrix")

from resources.trial_class import Trial
from TreadMetrix.data_postprocessing import process as post_processing
from TreadMetrix.ik_computing import process as compute_ik
from TreadMetrix.id_computing import process as compute_id
from TreadMetrix.joint_power_computing import process as compute_jp
from TreadMetrix.hip_joint_computation import compute_hip_joints

def time_normalize_df(df, points=101):
    """Normalizes a DataFrame to a fixed number of points using linear interpolation."""
    if df is None or len(df) < 2:
        return df
    
    # Check if 'time' or 'Time' exists (case insensitive)
    time_col = None
    for col in df.columns:
        if col.lower() == 'time':
            time_col = col
            break
    
    if time_col is None:
        # Assume first column is time if not labeled
        time_col = df.columns[0]
        
    # Original time indices (normalized to 0 to 1)
    t = df[time_col].values
    original_time = (t - t[0]) / (t[-1] - t[0])
    
    # Target time indices (0 to 1)
    new_time = np.linspace(0, 1, points)
    
    new_data = {'time_pct': new_time * 100}
    for col in df.columns:
        if col == time_col:
            continue
        # Interpolate
        try:
            f = interp1d(original_time, df[col], kind='linear', fill_value="extrapolate")
            new_data[col] = f(new_time)
        except Exception as e:
            # print(f"Warning: Could not interpolate column {col}: {e}")
            pass
        
    return pd.DataFrame(new_data)

def run_pipeline():
    mot_dir = r"Z:\AFO\Collected Data\P03\mot"
    trc_dir = r"Z:\AFO\Collected Data\P03\trc"
    model_file = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\P03-modle-scaled\Finalscaled.osim"
    output_base = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test"
    
    os.makedirs(output_base, exist_ok=True)
    
    # Matching MOT and TRC
    target_trial = "k6 speed test"
    m_file = os.path.join(mot_dir, target_trial + ".mot")
    t_file = os.path.join(trc_dir, target_trial + ".trc")
    
    if not os.path.exists(m_file) or not os.path.exists(t_file):
        print(f"Error: Could not find MOT or TRC for {target_trial}")
        return

    print(f"Processing Trial: {target_trial}")
    trial = Trial(mot=m_file, trc=t_file, name=target_trial)
    
    # 0. HJC Computation
    print("Step 0: Computing Hip Joint Centers...")
    updated_trc = compute_hip_joints(t_file)
    trial.trc = updated_trc
    
    # 1. Post-processing (Segmentation)
    print("Step 1: Post-processing (Segmentation)...")
    corrected_mot_path = os.path.join(output_base, "corrected_mot")
    segmented_path = os.path.join(output_base, "segmented")
    os.makedirs(corrected_mot_path, exist_ok=True)
    os.makedirs(segmented_path, exist_ok=True)
    
    # Override frame selection to process everything
    import TreadMetrix.data_postprocessing as pp
    pp.selected_start = 0
    pp.selected_end = len(trial.grf.data)
    
    post_processing(trial, save_plot_path=corrected_mot_path, 
                    save_segmented_path=segmented_path, 
                    show=False, save_optionals=True)
    
    # 2. IK
    print("Step 2: Inverse Kinematics...")
    ik_results_path = os.path.join(output_base, "ik_results")
    compute_ik(trial, model_file, ik_results_path, save=True)
    
    # 3. ID
    print("Step 3: Inverse Dynamics...")
    external_loads_path = os.path.join(output_base, "external_loads")
    id_results_path = os.path.join(output_base, "id_results")
    compute_id(trial, external_loads_path, id_results_path, model_file)
    
    # 4. Joint Power and Normalization
    print("Step 4: Joint Power with Time Normalization...")
    power_path = os.path.join(output_base, "joint_power")
    compute_jp(trial, power_path)
    
    # Now time-normalize the results and overwrite
    print("Time-normalizing joint power results...")
    for side in ["Right", "Left"]:
        side_power_path = os.path.join(power_path, side)
        for cycle in trial.gait_cycles[side]:
            if cycle.jp is not None:
                norm_df = time_normalize_df(cycle.jp.joint_power)
                out_name = f"{trial.name}_{side}_{cycle.num}_normalized.csv"
                out_path = os.path.join(side_power_path, out_name)
                norm_df.to_csv(out_path, index=False)
                
    print(f"Trial {target_trial} processed successfully.")
    
    # Visualization for verification
    visualize_results(trial, target_trial)

def visualize_results(trial, name):
    print("Generating verification plot...")
    sides = ["Right", "Left"]
    fig, axes = plt.subplots(len(sides), 2, figsize=(15, 10))
    
    for i, side in enumerate(sides):
        cycles = trial.gait_cycles[side]
        if not cycles:
            print(f"No cycles found for {side} side.")
            continue
            
        print(f"Plotting {len(cycles)} cycles for {side} side...")
        for cycle in cycles:
            if cycle.ik is not None:
                angle_col = 'ankle_angle_r' if side == "Right" else 'ankle_angle_l'
                if angle_col in cycle.ik.data.columns:
                    norm_angle = time_normalize_df(cycle.ik.data)
                    axes[i, 0].plot(norm_angle['time_pct'], norm_angle[angle_col], alpha=0.5)
            
            if cycle.id is not None:
                moment_col = 'ankle_angle_r_moment' if side == "Right" else 'ankle_angle_l_moment'
                if moment_col in cycle.id.data.columns:
                    norm_moment = time_normalize_df(cycle.id.data)
                    axes[i, 1].plot(norm_moment['time_pct'], norm_moment[moment_col], alpha=0.5)
                    
        axes[i, 0].set_title(f"{side} Ankle Angle (Normalized)")
        axes[i, 0].set_xlabel("% Gait Cycle")
        axes[i, 0].set_ylabel("Angle (deg)")
        
        axes[i, 1].set_title(f"{side} Ankle Moment (Normalized)")
        axes[i, 1].set_xlabel("% Gait Cycle")
        axes[i, 1].set_ylabel("Moment (Nm)")

    plt.tight_layout()
    plot_path = f"{name}_verification_plot.png"
    plt.savefig(plot_path)
    print(f"Verification plot saved to {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
