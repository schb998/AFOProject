import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resources.file_types.mot import MOT


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def read_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    needed = {"trial", "side", "start_plate"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Manifest missing columns {needed - set(df.columns)}")
    return df


def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    if data.size == 0:
        return data
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return data
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def ik_angle_columns_for_side(cols: list[str], side: str) -> list[str]:
    results = ["pelvis_tilt", "pelvis_list", "pelvis_rotation"]
    suffix = "r" if side == "Right" else "l"
    for c in cols:
        if c in ("time", "Time"):
            continue
        if len(c) >= 2 and c[-2] == "_" and c[-1].lower() == suffix:
            results.append(c)
    return [c for c in results if c in cols]


def id_moment_columns_for_side(cols: list[str], side: str) -> list[str]:
    # Keep your existing convention: *_moment columns + pelvis moments
    pattern = r"_r_" if side == "Right" else r"_l_"
    regex = re.compile(pattern)
    results = ["pelvis_tilt_moment", "pelvis_list_moment", "pelvis_rotation_moment"]
    for c in cols:
        if c.endswith("_moment") and regex.search(c):
            results.append(c)
    return [c for c in results if c in cols]


def compute_omega(ik_df: pd.DataFrame, angle_cols: list[str], cutoff_hz: float = 8.0) -> pd.DataFrame:
    t = ik_df["time"].to_numpy(dtype=float)
    fs = 1.0 / np.mean(np.diff(t))
    angles_deg = ik_df[angle_cols].to_numpy(dtype=float)
    angles_rad = angles_deg * (np.pi / 180.0)
    omega = np.gradient(angles_rad, t, axis=0)  # rad/s
    omega = butter_lowpass(omega, cutoff=cutoff_hz, fs=fs, order=4)
    out = pd.DataFrame(omega, columns=angle_cols)
    out.insert(0, "time", t)
    return out


def compute_power(omega_df: pd.DataFrame, id_df: pd.DataFrame, angle_cols: list[str], moment_cols: list[str]) -> pd.DataFrame:
    t_ik = omega_df["time"].to_numpy(dtype=float)
    t_id = id_df["time"].to_numpy(dtype=float)

    t0 = max(t_ik[0], t_id[0])
    t1 = min(t_ik[-1], t_id[-1])
    if t1 <= t0:
        raise ValueError("No overlap time between IK and ID")

    ik_mask = (t_ik >= t0) & (t_ik <= t1)
    id_mask = (t_id >= t0) & (t_id <= t1)

    t_use = omega_df.loc[ik_mask, "time"].to_numpy(dtype=float)
    omega_use = omega_df.loc[ik_mask, angle_cols].to_numpy(dtype=float)

    moments = id_df.loc[id_mask, moment_cols].to_numpy(dtype=float)
    fs_id = 1.0 / np.mean(np.diff(t_id[id_mask]))
    moments = butter_lowpass(moments, cutoff=6.0, fs=fs_id, order=4)

    # interpolate moments onto IK time
    moments_interp = np.zeros((len(t_use), moments.shape[1]))
    for j in range(moments.shape[1]):
        moments_interp[:, j] = np.interp(t_use, t_id[id_mask], moments[:, j])

    moment_base = [c.replace("_moment", "") for c in moment_cols]
    omega_map = {angle_cols[i]: omega_use[:, i] for i in range(len(angle_cols))}
    moment_map = {moment_base[j]: moments_interp[:, j] for j in range(len(moment_base))}

    power_cols = []
    power_data = []
    for ang in angle_cols:
        if ang in moment_map:
            power_cols.append(f"{ang}_power")
            power_data.append(omega_map[ang] * moment_map[ang])

    if not power_cols:
        raise ValueError("No matching angle/moment pairs for power")

    power_mat = np.vstack(power_data).T
    out = pd.DataFrame(power_mat, columns=power_cols)
    out.insert(0, "time", t_use)
    return out


def time_normalise(df_time: pd.DataFrame, n_points: int = 101) -> pd.DataFrame:
    t = df_time["time"].to_numpy(dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    gc = (t - t0) / (t1 - t0) * 100.0
    gc_target = np.linspace(0.0, 100.0, n_points)
    out = pd.DataFrame({"gc_percent": gc_target})
    for col in df_time.columns:
        if col == "time":
            continue
        out[col] = np.interp(gc_target, gc, df_time[col].to_numpy(dtype=float))
    return out


def plot_power(power_time: pd.DataFrame, power_gc: pd.DataFrame, out_png: str, side: str):
    ankle = "ankle_angle_r_power" if side == "Right" else "ankle_angle_l_power"
    plt.figure(figsize=(10, 4))
    if ankle in power_time.columns:
        plt.plot(power_time["time"], power_time[ankle], label=f"{ankle} (time)")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.grid(True)
    plt.legend()
    plt.title("Ankle power (time)")
    safe_mkdir(os.path.dirname(out_png))
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_time.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    if ankle in power_gc.columns:
        plt.plot(power_gc["gc_percent"], power_gc[ankle], label=f"{ankle} (GC)")
    plt.xlabel("Gait cycle (%)")
    plt.ylabel("Power (W)")
    plt.grid(True)
    plt.legend()
    plt.title("Ankle power (0–100% gait cycle)")
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_gc.png"), dpi=200)
    plt.close()


def main():
    DATA_ROOT = r"D:\TestOverground\Overground"
    PARTICIPANT = "PLB_03"

    participant_root = os.path.join(DATA_ROOT, PARTICIPANT)
    processed_root = os.path.join(participant_root, "processed")

    ik_root = os.path.join(processed_root, "ik")
    id_root = os.path.join(processed_root, "id")
    power_root = os.path.join(processed_root, "power")
    plots_root = os.path.join(processed_root, "plots", "power")
    manifest_path = os.path.join(processed_root, "manifests", "overground_cycles_manifest.csv")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    dfm = read_manifest(manifest_path)

    n_ok = 0
    n_fail = 0

    for _, r in dfm.iterrows():
        trial = str(r["trial"])
        side = str(r["side"])
        plate = int(r["start_plate"])

        # Derive cycle name from GRF filename (stable)
        grf_path = str(r.get("grf_path", ""))
        if not grf_path or not os.path.exists(grf_path):
            continue
        cycle_name = os.path.splitext(os.path.basename(grf_path))[0]

        # IK (prefer raw)
        ik_path = os.path.join(ik_root, trial, side, f"FP{plate}", f"{cycle_name}_ik_raw.mot")
        if not os.path.exists(ik_path):
            ik_path = os.path.join(ik_root, trial, side, f"FP{plate}", f"{cycle_name}.mot")
        if not os.path.exists(ik_path):
            print(f"[POWER] Missing IK: {ik_path}")
            n_fail += 1
            continue

        id_path = os.path.join(id_root, trial, side, f"FP{plate}", f"{cycle_name}_id.mot")
        if not os.path.exists(id_path):
            print(f"[POWER] Missing ID: {id_path}")
            n_fail += 1
            continue

        try:
            ik_df = MOT.load_from_mot(ik_path).data
            id_df = MOT.load_from_mot(id_path).data
            if "time" not in ik_df.columns:
                if "Time" in ik_df.columns:
                    ik_df = ik_df.rename(columns={"Time": "time"})
            if "time" not in id_df.columns:
                if "Time" in id_df.columns:
                    id_df = id_df.rename(columns={"Time": "time"})

            angle_cols = ik_angle_columns_for_side(list(ik_df.columns), side)
            moment_cols = id_moment_columns_for_side(list(id_df.columns), side)

            omega_df = compute_omega(ik_df, angle_cols, cutoff_hz=8.0)
            power_time = compute_power(omega_df, id_df, angle_cols, moment_cols)
            power_gc = time_normalise(power_time, n_points=101)

            out_dir = os.path.join(power_root, trial, side, f"FP{plate}")
            safe_mkdir(out_dir)
            out_time = os.path.join(out_dir, f"{cycle_name}_power_time.csv")
            out_gc = os.path.join(out_dir, f"{cycle_name}_power_gc.csv")
            power_time.to_csv(out_time, index=False)
            power_gc.to_csv(out_gc, index=False)

            out_png = os.path.join(plots_root, trial, side, f"FP{plate}", f"{cycle_name}_power.png")
            plot_power(power_time, power_gc, out_png, side)

            n_ok += 1
            print(f"[POWER] Wrote: {out_gc}")
        except Exception as e:
            print(f"[POWER] FAILED: {trial}/{side}/FP{plate}/{cycle_name} -> {repr(e)}")
            n_fail += 1

    print(f"\n[Done] Power completed. OK={n_ok}, FAIL={n_fail}")


if __name__ == "__main__":
    main()
