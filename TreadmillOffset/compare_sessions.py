import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def main():
    day01_csv = r"d:\AFO_Codes\TreadmillOffset\Day01\treadmill_offsets_summary.csv"
    day02_csv = r"d:\AFO_Codes\TreadmillOffset\Day02\treadmill_offsets_summary.csv"
    output_dir = r"d:\AFO_Codes\TreadmillOffset\Comparison"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load data
    df1 = pd.read_csv(day01_csv)
    df2 = pd.read_csv(day02_csv)
    
    force_cols = [
        'ground_force4_vx', 'ground_force4_vy', 'ground_force4_vz',
        'ground_force5_vx', 'ground_force5_vy', 'ground_force5_vz'
    ]
    
    # We will write a text summary
    summary_path = os.path.join(output_dir, "session_comparison_summary.txt")
    summary_file = open(summary_path, "w")
    
    summary_file.write("=================================================================\n")
    summary_file.write("TREADMILL FORCEPLATE OFFSET COMPARISON: DAY 01 VS DAY 02\n")
    summary_file.write("=================================================================\n\n")
    
    summary_file.write(f"Day01 trials: {len(df1)} files. Speeds: {df1['Speed'].min()}-{df1['Speed'].max()} mph. Slopes: {df1['Slope'].min()}-{df1['Slope'].max()}%\n")
    summary_file.write(f"Day02 trials: {len(df2)} files. Speeds: {df2['Speed'].min()}-{df2['Speed'].max()} mph. Slopes: {df2['Slope'].min()}-{df2['Slope'].max()}%\n\n")
    
    # ----------------------------------------------------
    # SECTION 1: COMPARISON AT THE OVERLAPPING SLOPE (3.1%)
    # ----------------------------------------------------
    summary_file.write("-----------------------------------------------------------------\n")
    summary_file.write("SECTION 1: DIRECT VALUE COMPARISON AT COMMON SLOPE (3.1%)\n")
    summary_file.write("-----------------------------------------------------------------\n")
    
    # Filter for slope = 3.1 in both days
    # Allow a small float tolerance (e.g. 3.0 to 3.2)
    s1_31 = df1[np.isclose(df1['Slope'], 3.1, atol=0.05)]
    s2_31 = df2[np.isclose(df2['Slope'], 3.1, atol=0.05)]
    
    # Find overlapping speeds
    shared_speeds = sorted(list(set(s1_31['Speed']).intersection(set(s2_31['Speed']))))
    summary_file.write(f"Shared speeds at 3.1% Slope: {shared_speeds}\n\n")
    
    # We will compute differences for median values
    differences_summary = []
    
    for col in force_cols:
        col_median = f"{col}_median"
        summary_file.write(f"Force Column: {col_median}\n")
        summary_file.write(f"{'Speed (mph)':<15}{'Day 01 Offset (N)':<20}{'Day 02 Offset (N)':<20}{'Difference (N)':<15}\n")
        summary_file.write("-" * 70 + "\n")
        
        abs_diffs = []
        for spd in shared_speeds:
            val1 = s1_31[np.isclose(s1_31['Speed'], spd, atol=0.01)][col_median].values[0]
            val2 = s2_31[np.isclose(s2_31['Speed'], spd, atol=0.01)][col_median].values[0]
            diff = val2 - val1
            abs_diffs.append(abs(diff))
            summary_file.write(f"{spd:<15.2f}{val1:<20.4f}{val2:<20.4f}{diff:<15.4f}\n")
            
        mean_abs_diff = np.mean(abs_diffs)
        max_abs_diff = np.max(abs_diffs)
        summary_file.write(f"Mean Absolute Difference: {mean_abs_diff:.4f} N\n")
        summary_file.write(f"Max Absolute Difference:  {max_abs_diff:.4f} N\n\n")
        
        differences_summary.append({
            'col': col,
            'mean_abs_diff': mean_abs_diff,
            'max_abs_diff': max_abs_diff
        })
        
    # ----------------------------------------------------
    # SECTION 2: OLS REGRESSION COEFFICIENT COMPARISON
    # ----------------------------------------------------
    summary_file.write("\n-----------------------------------------------------------------\n")
    summary_file.write("SECTION 2: REGRESSION COEFFICIENT COMPARISON\n")
    summary_file.write("Model: Offset = const + coef_speed * Speed + coef_slope * Slope\n")
    summary_file.write("-----------------------------------------------------------------\n\n")
    
    regression_comparison = []
    
    for col in force_cols:
        col_median = f"{col}_median"
        
        # Fit Day 01
        X1 = df1[['Speed', 'Slope']]
        X1 = sm.add_constant(X1)
        y1 = df1[col_median]
        model1 = sm.OLS(y1, X1).fit()
        
        # Fit Day 02
        X2 = df2[['Speed', 'Slope']]
        X2 = sm.add_constant(X2)
        y2 = df2[col_median]
        model2 = sm.OLS(y2, X2).fit()
        
        summary_file.write(f"=== {col} (Median) ===\n")
        summary_file.write(f"{'Parameter':<15}{'Day 01':<15}{'Day 02':<15}{'Difference':<15}\n")
        summary_file.write("-" * 60 + "\n")
        for param in ['const', 'Speed', 'Slope']:
            val1 = model1.params[param]
            val2 = model2.params[param]
            diff = val2 - val1
            summary_file.write(f"{param:<15}{val1:<15.4f}{val2:<15.4f}{diff:<15.4f}\n")
            
        summary_file.write(f"R-squared:     Day 01 = {model1.rsquared:.4f}, Day 02 = {model2.rsquared:.4f}\n")
        summary_file.write(f"F-pvalue:      Day 01 = {model1.f_pvalue:.4e}, Day 02 = {model2.f_pvalue:.4e}\n\n")
        
        regression_comparison.append({
            'col': col,
            'd1_params': model1.params,
            'd2_params': model2.params,
            'd1_r2': model1.rsquared,
            'd2_r2': model2.rsquared
        })
        
        # ----------------------------------------------------
        # PLOTTING COMPARATIVE GRAPHS AT SLOPE = 3.1%
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot raw points
        ax.scatter(s1_31['Speed'], s1_31[col_median], color='blue', marker='o', alpha=0.7, label='Day 01 Data (3.1% Slope)')
        ax.scatter(s2_31['Speed'], s2_31[col_median], color='red', marker='s', alpha=0.7, label='Day 02 Data (3.1% Slope)')
        
        # Plot regression lines evaluated at Slope = 3.1%
        # Create a range of speeds from min of both to max of both
        min_spd = min(df1['Speed'].min(), df2['Speed'].min())
        max_spd = max(df1['Speed'].max(), df2['Speed'].max())
        speeds_line = np.linspace(min_spd, max_spd, 100)
        
        y_pred_d1 = model1.params['const'] + model1.params['Speed'] * speeds_line + model1.params['Slope'] * 3.1
        y_pred_d2 = model2.params['const'] + model2.params['Speed'] * speeds_line + model2.params['Slope'] * 3.1
        
        ax.plot(speeds_line, y_pred_d1, color='blue', linestyle='--', label='Day 01 Fit (at 3.1% Slope)')
        ax.plot(speeds_line, y_pred_d2, color='red', linestyle='-.', label='Day 02 Fit (at 3.1% Slope)')
        
        ax.set_xlabel('Speed (mph)')
        ax.set_ylabel('Force Offset (N)')
        ax.set_title(f'Comparison: {col} Offset (Day01 vs Day02) at 3.1% Slope')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"plot_comparison_{col}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
    summary_file.close()
    print("Comparison analysis complete. Results saved in:", output_dir)

if __name__ == "__main__":
    main()
