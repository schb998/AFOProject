import sys
import os
import pandas as pd
import numpy as np

# Add TreadMetrix to sys.path so we can import offset_corrector
sys.path.insert(0, r"d:\AFO_Codes\TreadMetrix")
from offset_corrector import TreadmillOffsetCorrector

def main():
    print("==================================================")
    # 1. Initialize Corrector (should load the pooled offsets by default)
    print("Initializing TreadmillOffsetCorrector...")
    corrector = TreadmillOffsetCorrector()
    
    # Load raw summaries for exact comparison verification
    df1 = pd.read_csv(r"d:\AFO_Codes\TreadmillOffset\Day01\treadmill_offsets_summary.csv")
    df2 = pd.read_csv(r"d:\AFO_Codes\TreadmillOffset\Day02\treadmill_offsets_summary.csv")
    df_pooled = pd.read_csv(r"d:\AFO_Codes\TreadmillOffset\pooled_treadmill_offsets.csv")
    
    print("\n--- Test 1: Overlapping Condition (Speed=0.2, Slope=3.1) ---")
    # This is a shared condition between both days
    col = 'ground_force4_vx'
    col_median = f"{col}_median"
    
    val1 = df1[(df1['Speed'] == 0.2) & (df1['Slope'] == 3.1)][col_median].values[0]
    val2 = df2[(df2['Speed'] == 0.2) & (df2['Slope'] == 3.1)][col_median].values[0]
    expected_val = (val1 + val2) / 2
    
    # Retrieve offset from corrector
    offsets = corrector.get_offsets(speed=0.2, slope=3.1)
    retrieved_val = offsets[col]
    
    print(f"Day 01 Offset:  {val1:.6f} N")
    print(f"Day 02 Offset:  {val2:.6f} N")
    print(f"Expected Mean:  {expected_val:.6f} N")
    print(f"Retrieved Val:  {retrieved_val:.6f} N")
    
    assert np.isclose(retrieved_val, expected_val), "Error: Overlapping condition is not the average of both days!"
    print("Result: PASS (Overlapping condition correctly averaged!)")
    
    print("\n--- Test 2: Day 01 Unique Condition (Speed=1.0, Slope=0.0) ---")
    # This condition only exists on Day01
    val1_unique = df1[(df1['Speed'] == 1.0) & (df1['Slope'] == 0.0)][col_median].values[0]
    offsets_u1 = corrector.get_offsets(speed=1.0, slope=0.0)
    retrieved_u1 = offsets_u1[col]
    
    print(f"Day 01 Unique Offset: {val1_unique:.6f} N")
    print(f"Retrieved Offset:     {retrieved_u1:.6f} N")
    
    assert np.isclose(retrieved_u1, val1_unique), "Error: Day01 unique condition does not match!"
    print("Result: PASS (Day01 unique condition successfully retrieved!)")
    
    print("\n--- Test 3: Day 02 Unique Condition (Speed=1.0, Slope=7.0) ---")
    # This condition only exists on Day02
    val2_unique = df2[(df2['Speed'] == 1.0) & (df2['Slope'] == 7.0)][col_median].values[0]
    offsets_u2 = corrector.get_offsets(speed=1.0, slope=7.0)
    retrieved_u2 = offsets_u2[col]
    
    print(f"Day 02 Unique Offset: {val2_unique:.6f} N")
    print(f"Retrieved Offset:     {retrieved_u2:.6f} N")
    
    assert np.isclose(retrieved_u2, val2_unique), "Error: Day02 unique condition does not match!"
    print("Result: PASS (Day02 unique condition successfully retrieved!)")
    print("==================================================")
    print("All tests passed successfully! The pooled corrector model is correct.")

if __name__ == "__main__":
    main()
