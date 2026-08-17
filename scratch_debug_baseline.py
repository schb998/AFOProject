import pandas as pd
import numpy as np
import sys
sys.path.insert(0, r'd:\AFO_Codes')
from resources.trial_class import Trial
from TreadMetrix.offset_corrector import TreadmillOffsetCorrector
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

def filter_grf(mot_data, fs):
    b, a = butter(6, (12 / (fs / 2)), btype='low', output='ba')
    for col in mot_data.columns:
        mot_data[col] = filtfilt(b, a, mot_data[col])
    return mot_data

def baseline_correct_debug_test(mot_data, fz_col):
    fy = mot_data[fz_col]
    valley_indices, _ = find_peaks(-fy)
    if len(valley_indices) == 0:
        return "No valleys", 0
    min_v = np.min(fy[valley_indices])
    swing_valleys = valley_indices[fy[valley_indices] <= min_v + 50]
    if len(swing_valleys) == 0:
        return "No swing valleys", 0
    baseline = np.median(fy[swing_valleys])
    return baseline, min_v

trial = Trial(mot='Z:/AFO/Collected Data/P03-Processed/P03/P03/Old-AFO/mot/afo speed 0.mot')
fs = 1 / np.mean(np.diff(trial.grf.data['time']))
print(f"Sampling rate: {fs} Hz")

# Mock the offset corrector for a specific chunk (e.g. 20s to 40s)
corrector = TreadmillOffsetCorrector(summary_csv_path=r'd:\AFO_Codes\TreadmillOffset\treadmill_offsets_summary.csv')
offsets = corrector.get_offsets(speed=1.0, slope=0.0) # assuming speed 1.0 or whatever the user typed

df = trial.grf.data.copy()
mask = (df['time'] >= 22) & (df['time'] <= 38)
for col in corrector.force_cols:
    target_col = col
    if col not in df.columns:
        target_col = col.replace('4', '1').replace('5', '2')
    if target_col in df.columns:
        df.loc[mask, target_col] -= offsets[col]

df = filter_grf(df, fs)

b1, mv1 = baseline_correct_debug_test(df, 'ground_force1_vy')
b2, mv2 = baseline_correct_debug_test(df, 'ground_force2_vy')

print(f"Left Fy (1): min valley {mv1:.2f}, calculated baseline {b1:.2f}")
print(f"Right Fy (2): min valley {mv2:.2f}, calculated baseline {b2:.2f}")
