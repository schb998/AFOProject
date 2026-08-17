import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Get directories from command line arguments if available
data_dir = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\schb998\PhD\Data Backup\MyData\Treadmill Offset\Day01\MOT"
output_dir = sys.argv[2] if len(sys.argv) > 2 else r"d:\AFO_Codes\TreadmillOffset\Day01"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define relevant columns
force_cols = [
    'ground_force4_vx', 'ground_force4_vy', 'ground_force4_vz',
    'ground_force5_vx', 'ground_force5_vy', 'ground_force5_vz'
]

results = []

mot_files = glob.glob(os.path.join(data_dir, "*.mot"))
for filepath in mot_files:
    filename = os.path.basename(filepath)
    
    # Extract slope and speed
    slope_match = re.search(r'Slope_(\d+_\d+)', filename)
    speed_match = re.search(r'Speed_(\d+_\d+)', filename)
    
    if not speed_match:
        print(f"Skipping {filename}: No speed found.")
        continue
        
    slope = float(slope_match.group(1).replace('_', '.')) if slope_match else 0.0
    speed = float(speed_match.group(1).replace('_', '.'))
    
    # Read header to find endheader
    skip_rows = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() == 'endheader':
                skip_rows = i + 1
                break
                
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=skip_rows, index_col=False)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue
        
    # Check if columns exist
    missing_cols = [col for col in force_cols if col not in df.columns]
    if missing_cols:
        print(f"Skipping {filename}: Missing columns {missing_cols}")
        continue
        
    # Compute mean and median
    means = df[force_cols].mean()
    medians = df[force_cols].median()
    
    res = {
        'Filename': filename,
        'Slope': slope,
        'Speed': speed
    }
    
    for col in force_cols:
        res[f'{col}_mean'] = means[col]
        res[f'{col}_median'] = medians[col]
        
    results.append(res)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "treadmill_offsets_summary.csv"), index=False)

print(f"Processed {len(results_df)} files.")

# Perform Regression and Plotting
stats_out = open(os.path.join(output_dir, "regression_summary.txt"), "w")

X = results_df[['Speed', 'Slope']]
X = sm.add_constant(X)

for stat_type in ['mean', 'median']:
    for col in force_cols:
        y_col = f'{col}_{stat_type}'
        y = results_df[y_col]
        
        model = sm.OLS(y, X).fit()
        stats_out.write(f"\n{'='*50}\n")
        stats_out.write(f"Regression for {y_col}\n")
        stats_out.write(f"{'='*50}\n")
        stats_out.write(model.summary().as_text())
        stats_out.write("\n")
        
        # Create 3D Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(results_df['Speed'], results_df['Slope'], y, c='b', marker='o', label='Actual Data')
        
        # Create meshgrid for regression plane
        speed_range = np.linspace(results_df['Speed'].min(), results_df['Speed'].max(), 10)
        slope_range = np.linspace(results_df['Slope'].min(), results_df['Slope'].max(), 10)
        Speed_mesh, Slope_mesh = np.meshgrid(speed_range, slope_range)
        
        # Plane equation: y = beta0 + beta1*Speed + beta2*Slope
        Z_mesh = model.params['const'] + model.params['Speed'] * Speed_mesh + model.params['Slope'] * Slope_mesh
        
        ax.plot_surface(Speed_mesh, Slope_mesh, Z_mesh, alpha=0.5, color='r', label='Regression Plane')
        
        ax.set_xlabel('Speed (mph)')
        ax.set_ylabel('Slope (%)')
        ax.set_zlabel('Force Offset')
        ax.set_title(f'3D Regression: {y_col}')
        
        # Save plot
        plot_path = os.path.join(output_dir, f"plot_3d_{y_col}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Also create a 2D plot comparing Speed vs Offset (colored by Slope)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        scatter = ax2.scatter(results_df['Speed'], y, c=results_df['Slope'], cmap='viridis', label='Data')
        plt.colorbar(scatter, label='Slope (%)')
        
        # Add regression lines for a few fixed slopes
        speeds_line = np.linspace(results_df['Speed'].min(), results_df['Speed'].max(), 10)
        for s in [0.0, 2.5, 4.5]:
            y_pred = model.params['const'] + model.params['Speed'] * speeds_line + model.params['Slope'] * s
            ax2.plot(speeds_line, y_pred, linestyle='--', label=f'Fit (Slope {s}%)')
            
        ax2.set_xlabel('Speed (mph)')
        ax2.set_ylabel('Force Offset')
        ax2.set_title(f'2D Regression: {y_col}')
        ax2.legend()
        
        plot_path_2d = os.path.join(output_dir, f"plot_2d_{y_col}.png")
        plt.savefig(plot_path_2d)
        plt.close(fig2)

stats_out.close()
print("Analysis complete. Results and plots saved to:", output_dir)
