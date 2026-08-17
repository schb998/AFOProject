import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def time_normalize_df(df, points=101):
    if df is None or len(df) < 2: return df
    time_col = df.columns[0]
    t = df[time_col].values
    original_time = (t - t[0]) / (t[-1] - t[0])
    new_time = np.linspace(0, 1, points)
    new_data = {'time_pct': new_time * 100}
    for col in df.columns:
        if col == time_col: continue
        f = interp1d(original_time, df[col], kind='linear', fill_value="extrapolate")
        new_data[col] = f(new_time)
    return pd.DataFrame(new_data)

def generate_plot():
    output_base = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test"
    ik_path = os.path.join(output_base, "ik_results")
    id_path = os.path.join(output_base, "id_results")
    
    sides = ["Right", "Left"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, side in enumerate(sides):
        side_ik = os.path.join(ik_path, side)
        side_id = os.path.join(id_path, side)
        
        if not os.path.exists(side_ik): continue
        
        files = [f for f in os.listdir(side_ik) if f.endswith(".mot")]
        for f in files[:5]: # Plot first 5 cycles
            # IK
            ik_df = pd.read_csv(os.path.join(side_ik, f), sep='\t', skiprows=10) # OpenSim MOT header skip
            angle_col = 'ankle_angle_r' if side == "Right" else 'ankle_angle_l'
            if angle_col in ik_df.columns:
                norm_ik = time_normalize_df(ik_df)
                axes[i, 0].plot(norm_ik['time_pct'], norm_ik[angle_col], alpha=0.5)
            
            # ID
            id_file = f.replace(".mot", ".mot") # Filenames match in this case
            id_file_path = os.path.join(side_id, id_file)
            if os.path.exists(id_file_path):
                id_df = pd.read_csv(id_file_path, sep='\t', skiprows=10)
                moment_col = 'ankle_angle_r_moment' if side == "Right" else 'ankle_angle_l_moment'
                if moment_col in id_df.columns:
                    norm_id = time_normalize_df(id_df)
                    axes[i, 1].plot(norm_id['time_pct'], norm_id[moment_col], alpha=0.5)

        axes[i, 0].set_title(f"{side} Ankle Angle")
        axes[i, 0].set_ylabel("Angle (deg)")
        axes[i, 1].set_title(f"{side} Ankle Moment")
        axes[i, 1].set_ylabel("Moment (Nm)")
        
    plt.tight_layout()
    plt.savefig("p03_k6_verification.png")
    print("Verification plot saved to p03_k6_verification.png")

if __name__ == "__main__":
    generate_plot()
