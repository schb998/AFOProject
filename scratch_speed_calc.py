import pandas as pd
import numpy as np
from scipy.signal import find_peaks

file_path = r'Z:/AFO/Collected Data/P03-Processed/P03/P03/Old-AFO/mot/afo speed 0.mot'
df = pd.read_csv(file_path, sep='\t', skiprows=6)

# Filter for the 20s to 40s window
mask = (df['time'] >= 22) & (df['time'] <= 38)
df_sub = df[mask].copy()

# For speed, we need to look at the CoP during the stance phase.
# Stance phase is when vertical force (Vy) is high.
fy = df_sub['ground_force2_vy']
time = df_sub['time']
pz = df_sub['ground_force2_pz']
px = df_sub['ground_force2_px']

# We can calculate the speed by taking the derivative of CoP during stance.
# Let's find periods where fy > 300N
stance_mask = fy > 300

# Instead of fitting one line through all stance phases (they are separated by swing phases),
# we should isolate individual stance phases and calculate the slope of CoP.
diff_mask = np.diff(stance_mask.astype(int))
starts = np.where(diff_mask == 1)[0] + 1
ends = np.where(diff_mask == -1)[0]

# Make sure starts and ends align
if ends[0] < starts[0]:
    ends = ends[1:]
if starts[-1] > ends[-1]:
    starts = starts[:-1]

pz_speeds = []
px_speeds = []

for s, e in zip(starts, ends):
    if e - s > 10: # at least 10 frames of stance
        t_stance = time.iloc[s:e].values
        pz_stance = pz.iloc[s:e].values
        px_stance = px.iloc[s:e].values
        
        # Fit line
        pz_slope = np.polyfit(t_stance, pz_stance, 1)[0]
        px_slope = np.polyfit(t_stance, px_stance, 1)[0]
        
        pz_speeds.append(pz_slope)
        px_speeds.append(px_slope)

print(f"Average Pz speed: {np.mean(pz_speeds):.3f} m/s")
print(f"Average Px speed: {np.mean(px_speeds):.3f} m/s")

# Let's also print the average offset during the swing phase for this time window.
swing_mask = fy < 50
print(f"Right Plate (2) mean offset (Fy < 50N): {np.mean(fy[swing_mask]):.2f} N")
print(f"Left Plate (1) mean offset (Fy < 50N): {np.mean(df_sub['ground_force1_vy'][df_sub['ground_force1_vy'] < 150]):.2f} N")
