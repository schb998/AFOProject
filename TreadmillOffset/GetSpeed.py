import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ==================================================
# FILE PATH
# ==================================================

mot_file = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\Old-AFO\mot\afo speed 0.mot"


# ==================================================
# SETTINGS
# ==================================================

# In your OpenSim/Nexus exported files:
# Y is usually vertical
# Z is usually anterior-posterior / forward-backward
AP_AXIS = "z"

# Set to True only if CoP is in mm.
# OpenSim .mot files are usually in metres.
COP_IN_MM = False

# Force threshold for stance detection
force_threshold = 20  # N

# Leave as None for automatic detection
start_frame = None
end_frame = None


# ==================================================
# FUNCTIONS
# ==================================================

def read_mot_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    header_line_index = None

    for i, line in enumerate(lines):
        if line.strip().lower().startswith("time"):
            header_line_index = i
            break

    if header_line_index is None:
        raise ValueError("Could not find the data header line starting with 'time'.")

    df = pd.read_csv(
        file_path,
        sep=r"\s+|\t+",
        engine="python",
        skiprows=header_line_index
    )

    return df


def find_cop_columns(df):
    cop_groups = {}

    for col in df.columns:
        lower_col = col.lower()

        match = re.search(r"(.+)_p([xyz])$", lower_col)
        if match:
            prefix = match.group(1)
            axis = match.group(2)

            if prefix not in cop_groups:
                cop_groups[prefix] = {}

            cop_groups[prefix][axis] = col

    complete_groups = {
        prefix: axes for prefix, axes in cop_groups.items()
        if all(axis in axes for axis in ["x", "y", "z"])
    }

    return complete_groups


def find_vertical_force_columns(df):
    vertical_force_cols = []

    for col in df.columns:
        lower_col = col.lower()

        # Common OpenSim names:
        # ground_force_vy, ground_force_1_vy, 1_ground_force_vy, etc.
        if re.search(r"_v[y]$", lower_col):
            vertical_force_cols.append(col)

    return vertical_force_cols


def find_contact_windows(df, force_col, threshold=20):
    force = df[force_col].to_numpy()
    contact = force > threshold

    windows = []
    in_contact = False
    start = None

    for i, value in enumerate(contact):
        if value and not in_contact:
            start = i
            in_contact = True

        elif not value and in_contact:
            end = i - 1
            windows.append((start, end))
            in_contact = False

    if in_contact:
        windows.append((start, len(contact) - 1))

    return windows


def estimate_speed_from_cop(df, cop_col, start_frame, end_frame, cop_in_mm=False):
    selected = df.iloc[start_frame:end_frame + 1].copy()

    time = selected["time"].to_numpy()
    cop_ap = selected[cop_col].to_numpy()

    if cop_in_mm:
        cop_ap = cop_ap / 1000.0

    valid = np.isfinite(time) & np.isfinite(cop_ap)
    time = time[valid]
    cop_ap = cop_ap[valid]

    if len(time) < 3:
        raise ValueError("Not enough valid points in selected window.")

    result = linregress(time, cop_ap)

    signed_slope = result.slope
    speed_m_s = abs(signed_slope)
    speed_km_h = speed_m_s * 3.6

    return {
        "signed_slope_m_s": signed_slope,
        "speed_m_s": speed_m_s,
        "speed_km_h": speed_km_h,
        "r_squared": result.rvalue ** 2,
        "start_time": time[0],
        "end_time": time[-1],
        "duration": time[-1] - time[0],
        "cop_start": cop_ap[0],
        "cop_end": cop_ap[-1],
        "cop_displacement": cop_ap[-1] - cop_ap[0],
    }


def plot_signal(df, col, start_frame=None, end_frame=None, title=""):
    frames = np.arange(len(df))

    plt.figure(figsize=(12, 5))
    plt.plot(frames, df[col], linewidth=1.5)

    if start_frame is not None and end_frame is not None:
        plt.axvspan(start_frame, end_frame, alpha=0.25)

    plt.xlabel("Frame")
    plt.ylabel(col)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==================================================
# MAIN
# ==================================================

df = read_mot_file(mot_file)

print("\nLoaded file:")
print(mot_file)
print(f"Number of frames: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

cop_groups = find_cop_columns(df)

if not cop_groups:
    print("\nAvailable columns:")
    for col in df.columns:
        print(col)
    raise ValueError("No CoP columns were found. Check column names.")

print("\nDetected CoP groups:")
for i, (prefix, axes) in enumerate(cop_groups.items(), start=1):
    print(f"{i}. {prefix}")
    print(f"   px: {axes['x']}")
    print(f"   py: {axes['y']}")
    print(f"   pz: {axes['z']}")

# Use the first detected CoP group
selected_prefix = list(cop_groups.keys())[0]
cop_col = cop_groups[selected_prefix][AP_AXIS]

print(f"\nUsing CoP group: {selected_prefix}")
print(f"Using AP CoP column: {cop_col}")

vertical_force_cols = find_vertical_force_columns(df)

print("\nDetected vertical force columns:")
for col in vertical_force_cols:
    print(f"- {col}")

plot_signal(
    df,
    cop_col,
    title=f"Full CoP AP tracking: {cop_col}"
)

# Automatic frame selection
if start_frame is None or end_frame is None:

    if len(vertical_force_cols) == 0:
        raise ValueError(
            "No vertical GRF column found. Please manually set start_frame and end_frame."
        )

    force_col = vertical_force_cols[0]
    windows = find_contact_windows(df, force_col, threshold=force_threshold)

    if len(windows) == 0:
        raise ValueError("No contact windows found. Try lowering force_threshold.")

    print("\nDetected contact windows:")
    for i, (s, e) in enumerate(windows[:10], start=1):
        print(f"{i}. {s} to {e}, duration frames = {e - s + 1}")

    # Choose longest contact window
    hs, to = max(windows, key=lambda w: w[1] - w[0])

    # Use middle 30% of stance to reduce heel-strike and toe-off noise
    start_frame = int(hs + 0.35 * (to - hs))
    end_frame = int(hs + 0.65 * (to - hs))

    print("\nAutomatic selected window:")
    print(f"Contact window: {hs} to {to}")
    print(f"Selected CoP tracking frames: {start_frame} to {end_frame}")

plot_signal(
    df,
    cop_col,
    start_frame=start_frame,
    end_frame=end_frame,
    title=f"Selected CoP AP tracking window: {cop_col}"
)

result = estimate_speed_from_cop(
    df,
    cop_col,
    start_frame,
    end_frame,
    cop_in_mm=COP_IN_MM
)

print("\n===================================")
print("Estimated walking speed from CoP")
print("===================================")
print(f"File: {os.path.basename(mot_file)}")
print(f"CoP column used: {cop_col}")
print(f"Selected frames: {start_frame} to {end_frame}")
print(f"Selected time: {result['start_time']:.4f} to {result['end_time']:.4f} s")
print(f"Duration: {result['duration']:.4f} s")
print(f"CoP start: {result['cop_start']:.4f} m")
print(f"CoP end: {result['cop_end']:.4f} m")
print(f"CoP displacement: {result['cop_displacement']:.4f} m")
print(f"Signed slope: {result['signed_slope_m_s']:.4f} m/s")
print(f"Estimated speed: {result['speed_m_s']:.4f} m/s")
print(f"Estimated speed: {result['speed_km_h']:.4f} km/h")
print(f"Linear fit R²: {result['r_squared']:.4f}")

if result["r_squared"] < 0.8:
    print("\nWARNING:")
    print("The CoP tracking window is not very linear.")
    print("Try manually selecting a cleaner short region of CoP movement.")