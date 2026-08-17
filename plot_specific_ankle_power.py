import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_specific_cycles():
    base_path = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test\power_filtered"
    
    # Requested cycles
    targets = {
        "Right": [1, 2, 3, 4, 5],
        "Left": [1, 2, 3, 5, 6]
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    for i, side in enumerate(["Right", "Left"]):
        side_path = os.path.join(base_path, side)
        ax = axes[i]
        
        for cycle_num in targets[side]:
            filename = f"k6 speed test_{side}_{cycle_num}_normalized.csv"
            filepath = os.path.join(side_path, filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                col = 'ankle_angle_r_power' if side == "Right" else 'ankle_angle_l_power'
                if col in df.columns:
                    ax.plot(df['time_pct'], df[col], label=f"Cycle {cycle_num}")
                else:
                    print(f"Column {col} not found in {filename}")
            else:
                print(f"File not found: {filename}")
        
        ax.set_title(f"{side} Ankle Power (Normalized)")
        ax.set_xlabel("% Gait Cycle")
        ax.set_ylabel("Power (W)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "p03_ankle_power_comparison.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    plot_specific_cycles()
