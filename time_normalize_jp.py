import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def time_normalize_df(df, points=101):
    """Normalizes a DataFrame to a fixed number of points using linear interpolation."""
    if df is None or len(df) < 2:
        return df
    
    # Identify time column
    time_col = None
    for col in df.columns:
        if col.lower() == 'time':
            time_col = col
            break
    
    if time_col is None:
        time_col = df.columns[0]
        
    t = df[time_col].values
    original_time = (t - t[0]) / (t[-1] - t[0])
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
            pass
        
    return pd.DataFrame(new_data)

def main():
    base_path = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test\power_filtered"
    if not os.path.exists(base_path):
        # Try alternate path seen in full_pipeline output
        base_path = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test\power_filtered_corrected"
        
    if not os.path.exists(base_path):
        print(f"Error: Could not find power directory at {base_path}")
        return

    print(f"Normalizing results in {base_path}...")
    
    for side in ["Right", "Left"]:
        side_path = os.path.join(base_path, side)
        if not os.path.exists(side_path):
            continue
            
        for file in os.listdir(side_path):
            if file.endswith(".csv") and "_normalized" not in file:
                filepath = os.path.join(side_path, file)
                df = pd.read_csv(filepath)
                norm_df = time_normalize_df(df)
                
                # Save as new file
                out_name = file.replace(".csv", "_normalized.csv")
                out_path = os.path.join(side_path, out_name)
                norm_df.to_csv(out_path, index=False)
                print(f"Generated: {out_name}")

if __name__ == "__main__":
    main()
