import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def main():
    data_dir_right = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test\power_filtered_corrected\Right"
    data_dir_left = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test\power_filtered_corrected\Left"
    output_path = r"C:\Users\schb998\.gemini\antigravity\brain\37140e7c-89ad-4be8-9fef-a5e32ad2e20c\artifacts\all_ankle_powers.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    right_files = glob.glob(os.path.join(data_dir_right, "*.csv"))
    left_files = glob.glob(os.path.join(data_dir_left, "*.csv"))
    
    # Plot Right
    for file in right_files:
        df = pd.read_csv(file)
        # Normalize to 0-100%
        percent = np.linspace(0, 100, len(df))
        axes[0].plot(percent, df['ankle_angle_r_power'], color='blue', alpha=0.5)
        
    axes[0].set_title('All Right Ankle Powers')
    axes[0].set_xlabel('Gait Cycle (%)')
    axes[0].set_ylabel('Power (Watts)')
    axes[0].grid(True)
    
    # Plot Left
    for file in left_files:
        df = pd.read_csv(file)
        # Normalize to 0-100%
        percent = np.linspace(0, 100, len(df))
        axes[1].plot(percent, df['ankle_angle_l_power'], color='red', alpha=0.5)
        
    axes[1].set_title('All Left Ankle Powers')
    axes[1].set_xlabel('Gait Cycle (%)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    main()
