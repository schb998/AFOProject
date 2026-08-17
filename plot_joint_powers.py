import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    data_dir = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\joint_power\Right"
    output_path = r"C:\Users\schb998\.gemini\antigravity\brain\37140e7c-89ad-4be8-9fef-a5e32ad2e20c\artifacts\joint_powers_first_5.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cycles = [0, 1, 2, 3, 4]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for i, cycle in enumerate(cycles):
        file_path = os.path.join(data_dir, f"P03_Gait01_Right_{cycle}.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        
        axes[0].plot(df['time'], df['hip_flexion_r_power'], label=f"Cycle {cycle}", color=colors[i])
        axes[1].plot(df['time'], df['knee_angle_r_power'], label=f"Cycle {cycle}", color=colors[i])
        axes[2].plot(df['time'], df['ankle_angle_r_power'], label=f"Cycle {cycle}", color=colors[i])
        
    axes[0].set_title('Hip Flexion Power (Right)')
    axes[0].set_ylabel('Power (W/kg)')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_title('Knee Angle Power (Right)')
    axes[1].set_ylabel('Power (W/kg)')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].set_title('Ankle Angle Power (Right)')
    axes[2].set_ylabel('Power (W/kg)')
    axes[2].set_xlabel('Time Normalized (%)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    main()
