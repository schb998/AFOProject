import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def normalize_to_100(data):
    percent = np.linspace(0, 100, len(data))
    return percent, data

def main():
    trial_name = "k6 speed test"
    base_dir = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test"
    
    ik_dir = os.path.join(base_dir, "ik_results", "Right")
    id_dir = os.path.join(base_dir, "id_results", "Right")
    power_dir = os.path.join(base_dir, "power_filtered_corrected", "Right")
    
    # We will also need to get the GRF data for the segmented cycles.
    # The segmented GRFs are in segmented/Right
    grf_dir = os.path.join(base_dir, "segmented", "Right")
    
    ik_files = sorted(glob.glob(os.path.join(ik_dir, "*.mot")))
    id_files = sorted(glob.glob(os.path.join(id_dir, "*.mot")))
    power_files = sorted(glob.glob(os.path.join(power_dir, "*.csv")))
    grf_files = sorted(glob.glob(os.path.join(grf_dir, "*.mot")))

    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # Read mot function
    def read_mot(file_path):
        with open(file_path, 'r') as file:
            for _ in range(6):
                next(file)
            data = pd.read_csv(file, sep=r'\s+')
        return data

    for i in range(len(power_files)):
        try:
            power = pd.read_csv(power_files[i])
            ik = read_mot(ik_files[i])
            id_data = read_mot(id_files[i])
            grf = read_mot(grf_files[i])
            
            # Plot GRF (Vertical Force)
            # Find the correct column for vertical force (usually ground_force_vy or something)
            v_col = [col for col in grf.columns if 'vy' in col or 'vy' in col.lower()]
            if v_col:
                p, d = normalize_to_100(grf[v_col[0]])
                axes[0].plot(p, d, alpha=0.5)
                
            # Plot IK
            p, d = normalize_to_100(ik['ankle_angle_r'])
            axes[1].plot(p, d, alpha=0.5)
            
            # Plot ID
            p, d = normalize_to_100(id_data['ankle_angle_r_moment'])
            axes[2].plot(p, d, alpha=0.5)
            
            # Plot Power
            p, d = normalize_to_100(power['ankle_angle_r_power'])
            axes[3].plot(p, d, alpha=0.5)
            
        except Exception as e:
            print(f"Error processing index {i}: {e}")

    axes[0].set_title('Vertical GRF (Overlaid)')
    axes[0].set_ylabel('Force (N)')
    axes[0].grid(True)
    
    axes[1].set_title('Ankle Angle - IK (Overlaid)')
    axes[1].set_ylabel('Angle (deg)')
    axes[1].grid(True)
    
    axes[2].set_title('Ankle Moment - ID (Overlaid)')
    axes[2].set_ylabel('Moment (Nm)')
    axes[2].grid(True)
    
    axes[3].set_title('Ankle Power (Overlaid)')
    axes[3].set_ylabel('Power (W)')
    axes[3].set_xlabel('Gait Cycle (%)')
    axes[3].grid(True)
    
    plt.tight_layout()
    output_path = r"d:\AFO_Codes\pipeline_variance_check.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    main()
