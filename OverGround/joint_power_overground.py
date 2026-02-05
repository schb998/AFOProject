import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    if data.size == 0:
        return data
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return data
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)

def read_opensim_mot(path: str) -> pd.DataFrame:
    """Robust reader for OpenSim .mot/.sto: finds 'endheader' then reads whitespace table."""
    with open(path, "r") as f:
        lines = f.readlines()

    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            start = i + 1
            break

    raw = "".join(lines[start:]).strip()
    if not raw:
        raise ValueError(f"No data found after header in {path}")

    df = pd.read_csv(pd.io.common.StringIO(raw), sep=r"\s+")
    if "time" not in df.columns and "Time" in df.columns:
        df = df.rename(columns={"Time": "time"})
    if "time" not in df.columns:
        raise KeyError(f"'time' column not found in {path}. Columns={list(df.columns)}")
    return df

def iter_overground_cycles(ik_root: str):
    """Yields (trial, side, fp, ik_path) from: ik_root/<Trial>/<Side>/<FPx>/*.mot"""
    for trial in os.listdir(ik_root):
        tdir = os.path.join(ik_root, trial)
        if not os.path.isdir(tdir):
            continue
        for side in ["Left", "Right"]:
            sdir = os.path.join(tdir, side)
            if not os.path.isdir(sdir):
                continue
            for fp in os.listdir(sdir):
                fpdir = os.path.join(sdir, fp)
                if not os.path.isdir(fpdir):
                    continue
                for f in os.listdir(fpdir):
                    if f.lower().endswith(".mot"):
                        yield trial, side, fp, os.path.join(fpdir, f)

def base_cycle_name(filename: str) -> str:
    """Strips suffixes so IK and ID can match."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"(_ik|_id)$", "", name, flags=re.IGNORECASE)
    return name


def id_moment_columns_for_side(id_cols: list[str], side: str) -> list[str]:
    if side not in ("Left", "Right"):
        raise ValueError("side must be Left or Right")

    pattern = r"_r_" if side == "Right" else r"_l_"
    regex = re.compile(pattern)

    results = ["pelvis_tilt_moment", "pelvis_list_moment", "pelvis_rotation_moment"]
    for c in id_cols:
        if c.endswith("_moment") and regex.search(c):
            results.append(c)
    return [c for c in results if c in id_cols]

def ik_angle_columns_for_side(ik_cols: list[str], side: str) -> list[str]:
    results = ["pelvis_tilt", "pelvis_list", "pelvis_rotation"]
    suffix = "r" if side == "Right" else "l"
    for c in ik_cols:
        if c in ("time", "Time"):
            continue
        if len(c) >= 2 and c[-2] == "_" and c[-1].lower() == suffix:
            results.append(c)
    return [c for c in results if c in ik_cols]


def compute_angular_velocity_from_ik(
    ik_df: pd.DataFrame,
    angle_cols: list[str],
    cutoff_hz: float = 8.0
) -> pd.DataFrame:
    t = ik_df["time"].to_numpy(dtype=float)
    if len(t) < 5:
        raise ValueError("IK time series too short to compute angular velocity")

    fs = 1.0 / np.mean(np.diff(t))

    angles_deg = ik_df[angle_cols].to_numpy(dtype=float)
    angles_rad = angles_deg * (np.pi / 180.0)

    # omega = dtheta/dt (rad/s)
    omega = np.gradient(angles_rad, t, axis=0)

    omega_f = butter_lowpass(omega, cutoff=cutoff_hz, fs=fs, order=4)

    out = pd.DataFrame(omega_f, columns=angle_cols)
    out.insert(0, "time", t)
    return out

def compute_joint_power_time(
    omega_df: pd.DataFrame,
    id_df: pd.DataFrame,
    angle_cols: list[str],
    moment_cols: list[str],
    cutoff_moment_hz: float = 6.0
) -> pd.DataFrame:
    t_ik = omega_df["time"].to_numpy(dtype=float)
    t_id = id_df["time"].to_numpy(dtype=float)

    # overlap window
    t0 = max(t_ik[0], t_id[0])
    t1 = min(t_ik[-1], t_id[-1])
    if t1 <= t0:
        raise ValueError("No overlapping time range between IK and ID")

    ik_mask = (t_ik >= t0) & (t_ik <= t1)
    id_mask = (t_id >= t0) & (t_id <= t1)

    t_use = omega_df.loc[ik_mask, "time"].to_numpy(dtype=float)
    omega_use = omega_df.loc[ik_mask, angle_cols].to_numpy(dtype=float)

    moments = id_df.loc[id_mask, moment_cols].to_numpy(dtype=float)

    # filter moments in their native sampling
    if len(t_id[id_mask]) >= 5:
        fs_id = 1.0 / np.mean(np.diff(t_id[id_mask]))
        moments = butter_lowpass(moments, cutoff=cutoff_moment_hz, fs=fs_id, order=4)

    # interpolate moments onto IK time
    moments_interp = np.zeros((len(t_use), moments.shape[1]))
    for j in range(moments.shape[1]):
        moments_interp[:, j] = np.interp(t_use, t_id[id_mask], moments[:, j])

    moment_base = [c.replace("_moment", "") for c in moment_cols]

    # map for matching names
    omega_map = {angle_cols[i]: omega_use[:, i] for i in range(len(angle_cols))}
    moment_map = {moment_base[j]: moments_interp[:, j] for j in range(len(moment_base))}

    power_cols = []
    power_data = []
    for ang in angle_cols:
        if ang in moment_map:
            power_cols.append(f"{ang}_power")
            power_data.append(omega_map[ang] * moment_map[ang])

    if not power_cols:
        raise ValueError("No matching angle/moment pairs found to compute power")

    power_mat = np.vstack(power_data).T
    out = pd.DataFrame(power_mat, columns=power_cols)
    out.insert(0, "time", t_use)
    return out

def time_normalise_to_gait_cycle(df_time: pd.DataFrame, n_points: int = 100) -> pd.DataFrame:
    """
    Convert time-based dataframe (seconds) to 0–100% gait cycle.
    Uses linear interpolation onto evenly spaced percent points.

    Output columns:
      gc_percent, <signals...>
    """
    if "time" not in df_time.columns:
        raise KeyError("Expected a 'time' column for time-normalisation")

    t = df_time["time"].to_numpy(dtype=float)
    if len(t) < 2:
        raise ValueError("Not enough samples to time-normalise")

    # Percent axis from actual time
    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        raise ValueError("Invalid time range for normalisation")

    gc = (t - t0) / (t1 - t0) * 100.0
    gc_target = np.linspace(0.0, 100.0, n_points)

    out = pd.DataFrame({"gc_percent": gc_target})

    for col in df_time.columns:
        if col == "time":
            continue
        y = df_time[col].to_numpy(dtype=float)
        out[col] = np.interp(gc_target, gc, y)

    return out


def main():
    DATA_ROOT = r"D:\TestOverground\Overground"
    PARTICIPANT = "PLB_03"

    participant_root = os.path.join(DATA_ROOT, PARTICIPANT)
    processed_root = os.path.join(participant_root, "processed")

    ik_root = os.path.join(processed_root, "ik")
    id_root = os.path.join(processed_root, "id")
    power_root = os.path.join(processed_root, "power")

    if not os.path.isdir(ik_root):
        raise FileNotFoundError(f"IK folder not found: {ik_root}")
    if not os.path.isdir(id_root):
        raise FileNotFoundError(f"ID folder not found: {id_root}")

    total = 0
    written = 0

    for trial, side, fp, ik_path in iter_overground_cycles(ik_root):
        total += 1
        cycle_base = base_cycle_name(ik_path)

        id_dir = os.path.join(id_root, trial, side, fp)
        if not os.path.isdir(id_dir):
            print(f"[POWER] Missing ID folder: {id_dir}")
            continue

        candidates = [
            os.path.join(id_dir, f"{cycle_base}_id.mot"),
            os.path.join(id_dir, f"{cycle_base}_id.sto"),
            os.path.join(id_dir, f"{cycle_base}.mot"),
            os.path.join(id_dir, f"{cycle_base}.sto"),
        ]
        id_path = next((p for p in candidates if os.path.exists(p)), None)
        if id_path is None:
            print(f"[POWER] Missing ID file for {trial}/{side}/{fp}/{cycle_base}")
            continue

        try:
            ik_df = read_opensim_mot(ik_path)
            id_df = read_opensim_mot(id_path)

            angle_cols = ik_angle_columns_for_side(list(ik_df.columns), side)
            moment_cols = id_moment_columns_for_side(list(id_df.columns), side)

            omega_df = compute_angular_velocity_from_ik(ik_df, angle_cols, cutoff_hz=8.0)
            power_time = compute_joint_power_time(omega_df, id_df, angle_cols, moment_cols, cutoff_moment_hz=6.0)

            # Time-normalise to 0–100%
            power_gc = time_normalise_to_gait_cycle(power_time, n_points=101)

            out_dir = os.path.join(power_root, trial, side, fp)
            safe_mkdir(out_dir)

            out_time = os.path.join(out_dir, f"{cycle_base}_power_time.csv")
            out_gc = os.path.join(out_dir, f"{cycle_base}_power_gc.csv")

            power_time.to_csv(out_time, index=False)
            power_gc.to_csv(out_gc, index=False)

            written += 1

            ankle_col = "ankle_angle_r_power" if side == "Right" else "ankle_angle_l_power"
            peak = power_time[ankle_col].abs().max() if ankle_col in power_time.columns else np.nan
            print(f"[POWER] Wrote: {out_gc} (GC-normalised). Peak ankle power={peak:.2f} W")

        except Exception as e:
            print(f"[POWER] FAILED: {trial}/{side}/{fp}/{cycle_base} -> {repr(e)}")
            continue

    print(f"\n[Done] Processed {total} IK cycles, wrote {written} power outputs.")


if __name__ == "__main__":
    main()
