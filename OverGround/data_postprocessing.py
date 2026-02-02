import os
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.signal import butter, filtfilt

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
from resources.custom_exceptions import MissingPathException
from resources.trial_class import Trial, GaitCycle



def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def find_file_ignore_case(folder: str, filename: str) -> str | None:
    if not os.path.isdir(folder):
        return None
    target = filename.lower()
    for f in os.listdir(folder):
        if f.lower() == target:
            return os.path.join(folder, f)
    return None


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def nearest_index(time_vec: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(time_vec - t)))


def parse_side_cell(x: str) -> str:
    """Return 'left'/'right'/'both'/'none'."""
    val = str(x).strip().lower()
    if val in ("l", "left"):
        return "left"
    if val in ("r", "right"):
        return "right"
    if val in ("b", "both"):
        return "both"
    return "none"



# Infosheet

class OvergroundInfoSheet:
    """
    Expected headers (case-sensitive):
      - Trials/Events
      - ID                (1/0 or Yes/No)
      - Dynamic           (Yes/No)  (optional, but often present)
      - Valid GaitCycle   (Right/Left/Both/None)
      - FP1, FP2, FP3     (Left/Right/Both/blank)
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path).copy()

        needed = ["Trials/Events", "ID", "Valid GaitCycle", "FP1", "FP2", "FP3"]
        for c in needed:
            if c not in self.df.columns:
                raise KeyError(f"Infosheet missing required column: '{c}'")

        # normalize string columns
        for c in ["Trials/Events", "Valid GaitCycle", "FP1", "FP2", "FP3", "Dynamic", "ID"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).str.strip()

    def row_for_trial(self, trial_name: str) -> pd.Series:
        rows = self.df[self.df["Trials/Events"] == str(trial_name)]
        if rows.empty:
            rows2 = self.df[self.df["Trials/Events"].astype(str).str.lower() == str(trial_name).lower()]
            if rows2.empty:
                raise ValueError(f"Trial '{trial_name}' not found in infosheet.")
            return rows2.iloc[0]
        return rows.iloc[0]

    @staticmethod
    def is_yes(x: str) -> bool:
        return str(x).strip().lower() in ("yes", "y", "true", "1")

    def use_for_id(self, trial_name: str) -> bool:
        """Your rule: ID column must be 1/Yes."""
        r = self.row_for_trial(trial_name)
        return self.is_yes(r.get("ID", "0"))

    def valid_sides(self, trial_name: str) -> list[str]:
        r = self.row_for_trial(trial_name)
        val = str(r.get("Valid GaitCycle", "")).strip().lower()
        if val in ("right", "r"):
            return ["Right"]
        if val in ("left", "l"):
            return ["Left"]
        if val in ("both", "b"):
            return ["Right", "Left"]
        return []

    def fp_side_map(self, trial_name: str) -> dict[int, str]:
        r = self.row_for_trial(trial_name)
        out = {}
        for i in (1, 2, 3):
            out[i] = parse_side_cell(r.get(f"FP{i}", ""))
        return out

    def trial_names(self) -> list[str]:
        return self.df["Trials/Events"].astype(str).tolist()

# GRF filtering + baseline correction


def filter_grf(mot: MOT, fs: float, cutoff_hz: float = 12.0, order: int = 6) -> None:
    b, a = butter(order, (cutoff_hz / (fs / 2)), btype="low", output="ba")
    filtered_df = deepcopy(mot.data)
    for col in mot.data.columns.tolist():
        filtered_df[col] = filtfilt(b, a, mot.data[col])
    mot.data = filtered_df


def baseline_correct(mot_object: MOT, fz_col: str, related_cols: list[str]) -> None:
    if fz_col not in mot_object.data.columns:
        return

    fy = mot_object.data[fz_col].to_numpy()
    if len(fy) < 3:
        return

    corrected_df = deepcopy(mot_object.data)

    valley_idx = np.where((fy[1:-1] < fy[:-2]) & (fy[1:-1] < fy[2:]))[0] + 1
    swing_valleys = valley_idx[fy[valley_idx] < 0]
    if len(swing_valleys) == 0:
        return

    baseline = abs(np.median(fy[swing_valleys]))
    corrected_df[fz_col] = corrected_df[fz_col] + baseline

    for col in related_cols:
        if col not in corrected_df.columns:
            continue
        related = mot_object.data[col].to_numpy()
        offset = np.median(related[swing_valleys])
        corrected_df[col] = corrected_df[col] - offset if offset > 0 else corrected_df[col] + abs(offset)

    mot_object.data = corrected_df

# Contact detection (HS/TO from threshold)


def detect_contacts_threshold(vy: np.ndarray,
                             threshold: float,
                             min_contact_samples: int,
                             min_swing_samples: int) -> list[tuple[int, int]]:
    """
    Returns stance intervals [(hs_idx, to_idx), ...]
      HS: rising edge (vy > threshold)
      TO: falling edge
    """
    contact = vy > threshold
    edges = np.diff(contact.astype(int))

    hs = list(np.where(edges == 1)[0] + 1)
    to = list(np.where(edges == -1)[0] + 1)

    intervals = []
    j = 0
    for h in hs:
        while j < len(to) and to[j] < h:
            j += 1
        if j >= len(to):
            break
        t = to[j]
        if (t - h) >= min_contact_samples:
            intervals.append((h, t))

    cleaned = []
    prev_to = None
    for h, t in intervals:
        if prev_to is not None and (h - prev_to) < min_swing_samples:
            continue
        cleaned.append((h, t))
        prev_to = t

    return cleaned


def detect_overground_contacts(mot: MOT, fs: float, threshold: float = 20.0) -> dict[int, list[tuple[int, int]]]:
    out = {}
    for plate in (1, 2, 3):
        col = f"ground_force{plate}_vy"
        if col not in mot.data.columns:
            out[plate] = []
            continue
        vy = mot.data[col].to_numpy()
        out[plate] = detect_contacts_threshold(
            vy,
            threshold=threshold,
            min_contact_samples=int(0.05 * fs),
            min_swing_samples=int(0.05 * fs),
        )
    return out


#  Build GAIT CYCLES as HS -> next HS (per side)


def build_hs_events_by_side(mot: MOT,
                            contacts_by_plate: dict[int, list[tuple[int, int]]],
                            fp_map: dict[int, str]) -> dict[str, list[dict]]:
    """
    Returns:
      {
        "Right": [ {"plate":2, "hs_idx":..., "hs_time":...}, ... ],
        "Left":  [ ... ]
      }

    Collect HS events across all plates mapped to that side.
    """
    mot_t = mot.data["time"].to_numpy()

    events = {"Right": [], "Left": []}
    for plate, intervals in contacts_by_plate.items():
        side = fp_map.get(plate, "none")  # left/right/both/none
        if side not in ("left", "right"):
            continue  # ignore both/none
        side_key = "Right" if side == "right" else "Left"

        for (hs_idx, to_idx) in intervals:
            hs_time = float(mot_t[int(hs_idx)])
            events[side_key].append({
                "plate": int(plate),
                "hs_idx": int(hs_idx),
                "hs_time": hs_time,
                "to_idx": int(to_idx),
                "to_time": float(mot_t[int(to_idx)])
            })

    # sort events in time order
    for s in ["Right", "Left"]:
        events[s].sort(key=lambda d: d["hs_time"])

    return events


def segment_cycles_hs_to_hs(trial: Trial,
                            info: OvergroundInfoSheet,
                            contacts_by_plate: dict[int, list[tuple[int, int]]],
                            save_root: str,
                            pad_s: float = 0.05,
                            min_cycle_s: float = 0.30,
                            max_cycle_s: float = 2.50) -> list[dict]:
    """
    Creates cycles for each valid side:
      cycle i = HS_i -> HS_(i+1) for that side (across plates)

    The forceplate assigned to the cycle is the plate at HS_i (start event),
    e.g. Right HS on FP3 then next Right HS on FP1 => one Right cycle, plate=3.

    Uses ABSOLUTE FRAME sampling for MOT/TRC.
    """
    mot = trial.corrected_grf
    trc = trial.trc
    if trc is None:
        raise MissingPathException(f"Markers trajectory object (TRC) for trial {trial.name}", "No such object given.")

    valid_sides = info.valid_sides(trial.name)
    fp_map = info.fp_side_map(trial.name)

    print(f"[DEBUG] Valid sides: {valid_sides}")
    print(f"[DEBUG] FP map: {fp_map}")
    print(f"[DEBUG] Contacts per plate: { {p: len(iv) for p, iv in contacts_by_plate.items()} }")

    if len(valid_sides) == 0:
        return []

    mot_t = mot.data["time"].to_numpy()
    trc_t = trc.data["Time"].to_numpy()

    pad_s = float(pad_s)

    mot_first = int(getattr(mot, "first_frame", 0))
    trc_first = int(getattr(trc, "first_frame", 0))
    mot_n = mot.data.shape[0]
    trc_n = trc.data.shape[0]
    mot_abs_min = mot_first
    mot_abs_max = mot_first + mot_n - 1
    trc_abs_min = trc_first
    trc_abs_max = trc_first + trc_n - 1

    # build HS events per side
    hs_events = build_hs_events_by_side(mot, contacts_by_plate, fp_map)

    manifest_rows = []
    total_cycles = 0

    for side in ["Right", "Left"]:
        if side not in valid_sides:
            continue

        ev = hs_events.get(side, [])
        if len(ev) < 2:

            continue

        for i in range(len(ev) - 1):
            hs1 = ev[i]
            hs2 = ev[i + 1]

            start_time = float(hs1["hs_time"]) - pad_s
            end_time = float(hs2["hs_time"]) + pad_s

            duration = end_time - start_time
            if duration < min_cycle_s or duration > max_cycle_s:
                # avoid weird tiny segments or huge gaps
                continue

            # Convert times to indices
            mot_start_idx = nearest_index(mot_t, start_time)
            mot_end_idx = nearest_index(mot_t, end_time)
            trc_start_idx = nearest_index(trc_t, start_time)
            trc_end_idx = nearest_index(trc_t, end_time)

            # Clamp indices to array bounds
            mot_start_idx = clamp_int(mot_start_idx, 0, mot_n - 1)
            mot_end_idx = clamp_int(mot_end_idx, 0, mot_n - 1)
            trc_start_idx = clamp_int(trc_start_idx, 0, trc_n - 1)
            trc_end_idx = clamp_int(trc_end_idx, 0, trc_n - 1)

            if mot_end_idx <= mot_start_idx or trc_end_idx <= trc_start_idx:
                continue

            # Convert to ABSOLUTE frames
            mot_start_frame = mot_first + int(mot_start_idx)
            mot_end_frame = mot_first + int(mot_end_idx)
            trc_start_frame = trc_first + int(trc_start_idx)
            trc_end_frame = trc_first + int(trc_end_idx)

            # Clamp absolute frames
            mot_start_frame = clamp_int(mot_start_frame, mot_abs_min, mot_abs_max)
            mot_end_frame = clamp_int(mot_end_frame, mot_abs_min, mot_abs_max)
            trc_start_frame = clamp_int(trc_start_frame, trc_abs_min, trc_abs_max)
            trc_end_frame = clamp_int(trc_end_frame, trc_abs_min, trc_abs_max)

            if mot_end_frame <= mot_start_frame or trc_end_frame <= trc_start_frame:
                continue

            # Sample
            try:
                grf_seg = mot.sample(mot_start_frame, mot_end_frame)
                trc_seg = trc.sample(trc_start_frame, trc_end_frame)
            except Exception as e:
                print(f"[WARN] Skipping cycle {trial.name} {side} HS->HS sampling failed: {repr(e)}")
                print(f"       MOT frames: {mot_start_frame}..{mot_end_frame} (valid {mot_abs_min}..{mot_abs_max})")
                print(f"       TRC frames: {trc_start_frame}..{trc_end_frame} (valid {trc_abs_min}..{trc_abs_max})")
                continue

            cycle_num = len(trial.gait_cycles[side]) + 1
            cycle = GaitCycle(side=side, number=cycle_num)

            # IMPORTANT:
            # Use the plate of the FIRST HS as the plate for this gait cycle
            # (example: Right HS on FP3 then Right HS on FP1 => cycle plate = 3)
            cycle.forceplate_num = int(hs1["plate"])

            cycle.add_grf(grf_object=grf_seg)
            cycle.add_trc(trc_object=trc_seg)
            trial.gait_cycles[side].append(cycle)

            # Save
            cycle_dir = os.path.join(save_root, trial.name, side, f"FP{cycle.forceplate_num}", f"cycle_{cycle_num}")
            safe_mkdir(cycle_dir)

            grf_filename = f"{trial.name}_{side}_cycle{cycle_num}.mot"
            trc_filename = f"{trial.name}_{side}_cycle{cycle_num}.trc"

            grf_seg.rename(name=f"{trial.name}_{side.lower()}_cycle{cycle_num}", filename=grf_filename)
            trc_seg.rename(filename=trc_filename)

            grf_seg.save(cycle_dir)
            trc_seg.save(cycle_dir)

            cycle.grf.filepath = os.path.join(cycle_dir, grf_filename)
            cycle.trc.filepath = os.path.join(cycle_dir, trc_filename)

            manifest_rows.append({
                "trial": trial.name,
                "side": side,
                "cycle_num": cycle_num,
                "forceplate_start_hs": int(hs1["plate"]),
                "hs1_time": float(hs1["hs_time"]),
                "hs2_time": float(hs2["hs_time"]),
                "start_time": float(mot_t[mot_start_idx]),
                "end_time": float(mot_t[mot_end_idx]),
                "grf_path": cycle.grf.filepath,
                "trc_path": cycle.trc.filepath,
            })

            total_cycles += 1

    if total_cycles == 0:
        print(f"[WARN] No HS->HS gait cycles segmented for {trial.name}.")
        print("       Common reasons: only 1 HS in plates for that side, FP map marked as 'both/none', or time mismatch.")

    return manifest_rows


# Trial loader

def load_trial_objects(trial_name: str, grf_path: str, trc_path: str) -> Trial:
    # Trial requires mot. Passing paths works with your Trial.__init__
    return Trial(mot=grf_path, trc=trc_path, name=trial_name)


def process_overground_trial(trial: Trial,
                             info: OvergroundInfoSheet,
                             corrected_out: str,
                             segmented_out: str,
                             threshold: float = 20.0) -> list[dict]:

    fs = 1 / np.mean(np.diff(trial.grf.data["time"]))
    print(f"\n[GRF] Processing overground trial: {trial.name} @ {fs:.1f} Hz")

    corrected = trial.grf.copy()
    corrected.rename(name=trial.name, filename=f"{trial.name}.mot")

    filter_grf(corrected, fs)

    for plate in (1, 2, 3):
        vy = f"ground_force{plate}_vy"
        baseline_correct(corrected, vy, [f"ground_force{plate}_vx", f"ground_force{plate}_vz"])

    trial.add_corrected_grf(corrected_grf=corrected)

    safe_mkdir(corrected_out)
    corrected.save(corrected_out)

    # only segment trials where ID == 1
    if not info.use_for_id(trial.name):
        print(f"[INFO] Skipping segmentation for {trial.name}: infosheet ID != 1.")
        return []

    contacts = detect_overground_contacts(corrected, fs, threshold=threshold)

    safe_mkdir(segmented_out)
    return segment_cycles_hs_to_hs(
        trial=trial,
        info=info,
        contacts_by_plate=contacts,
        save_root=segmented_out,
        pad_s=0.05  # small pad so we don't cut too tight
    )


# MAIN


def main():
    # path
    DATA_ROOT = r"D:\TestOverground\Overground"
    PARTICIPANT = "PLB_03"
    INFO_CSV_NAME = "Trials_PLB_03.csv"
    CONTACT_THRESHOLD_N = 20.0

    participant_root = os.path.join(DATA_ROOT, PARTICIPANT)

    infosheet_path = os.path.join(participant_root, "infosheet", INFO_CSV_NAME)
    raw_grf_dir = os.path.join(participant_root, "raw", "grf_mot")
    raw_trc_dir = os.path.join(participant_root, "raw", "markers_trc")

    processed_root = os.path.join(participant_root, "processed")
    corrected_out = os.path.join(processed_root, "grf_corrected")
    segmented_out = os.path.join(processed_root, "segmented")
    manifests_out = os.path.join(processed_root, "manifests")
    safe_mkdir(manifests_out)

    manifest_path = os.path.join(manifests_out, "overground_cycles_manifest.csv")

    if not os.path.exists(infosheet_path):
        raise FileNotFoundError(f"Infosheet not found: {infosheet_path}")

    info = OvergroundInfoSheet(infosheet_path)

    all_rows = []

    for trial_name in info.trial_names():
        grf_file = find_file_ignore_case(raw_grf_dir, f"{trial_name}.mot")
        trc_file = find_file_ignore_case(raw_trc_dir, f"{trial_name}.trc")

        if grf_file is None or trc_file is None:
            print(f"[SKIP] {trial_name}: missing GRF/TRC file.")
            continue

        trial = load_trial_objects(trial_name, grf_file, trc_file)

        rows = process_overground_trial(
            trial=trial,
            info=info,
            corrected_out=corrected_out,
            segmented_out=segmented_out,
            threshold=CONTACT_THRESHOLD_N
        )

        for r in rows:
            r["participant"] = PARTICIPANT
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(manifest_path, index=False)

    print("\n[Done] GRF correction + HS->HS segmentation completed.")
    print(f"[Done] Manifest written: {manifest_path}")
    print(f"[Done] Total segmented cycles: {len(df)}")


if __name__ == "__main__":
    main()