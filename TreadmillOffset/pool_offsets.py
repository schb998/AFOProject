import os
import pandas as pd
import numpy as np

def main():
    day01_csv = r"d:\AFO_Codes\TreadmillOffset\Day01\treadmill_offsets_summary.csv"
    day02_csv = r"d:\AFO_Codes\TreadmillOffset\Day02\treadmill_offsets_summary.csv"
    output_csv = r"d:\AFO_Codes\TreadmillOffset\pooled_treadmill_offsets.csv"
    
    if not os.path.exists(day01_csv):
        raise FileNotFoundError(f"Day01 summary not found at: {day01_csv}")
    if not os.path.exists(day02_csv):
        raise FileNotFoundError(f"Day02 summary not found at: {day02_csv}")
        
    df1 = pd.read_csv(day01_csv)
    df2 = pd.read_csv(day02_csv)
    
    # Concatenate the two DataFrames
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # We want to group by Speed and Slope, and average all numeric offset columns
    # We first extract the numeric columns (ignoring Filename, Speed, and Slope)
    numeric_cols = [col for col in df_combined.columns if col not in ['Filename', 'Speed', 'Slope']]
    
    # Group by Speed and Slope and take the mean
    # This automatically handles:
    # - Overlapping conditions (averages the 2 values)
    # - Non-overlapping conditions (returns the single value)
    df_grouped = df_combined.groupby(['Speed', 'Slope'])[numeric_cols].mean().reset_index()
    
    # Re-create a clean Filename for each row based on Speed and Slope
    def make_filename(row):
        speed_str = str(row['Speed']).replace('.', '_')
        slope_str = str(row['Slope']).replace('.', '_')
        return f"Pooled_Slope_{slope_str}_Speed_{speed_str}.mot"
        
    df_grouped['Filename'] = df_grouped.apply(make_filename, axis=1)
    
    # Reorder columns to match original: Filename, Slope, Speed, then the rest
    other_cols = [col for col in df_grouped.columns if col not in ['Filename', 'Slope', 'Speed']]
    ordered_cols = ['Filename', 'Slope', 'Speed'] + other_cols
    df_grouped = df_grouped[ordered_cols]
    
    # Save the pooled offsets
    df_grouped.to_csv(output_csv, index=False)
    print(f"Successfully pooled Day01 and Day02 offsets!")
    print(f"Total rows in Day01: {len(df1)}")
    print(f"Total rows in Day02: {len(df2)}")
    print(f"Total rows in pooled file: {len(df_grouped)} (saved to: {output_csv})")

if __name__ == "__main__":
    main()
