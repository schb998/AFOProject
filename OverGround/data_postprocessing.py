import os
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.signal import butter, filtfilt

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
from resources.custom_exceptions import MissingPathException
from resources.trial_class import Trial, GaitCycle
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

# Utilities
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
    """
    Return role for the FP cell:
      - 'left' / 'right'  -> usable and side-known
      - 'unknown'         -> 'Both' (unusable for kinetics, but HS can be used as a boundary)
      - 'none'            -> blank/none
    """
    val = str(x).strip().lower()
    if val in ("l", "left"):
        return "left"
    if val in ("r", "right"):
        return "right"
    if val in ("b", "both"):
        return "unknown"
    return "none"


def to_yes_flag(x) -> bool:
    """Robust Yes/No parsing (for Dynamic)."""
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "t"):
        return True
    try:
        return float(s) != 0.0
    except ValueError:
        return False


def to_int_count(x) -> int:
    """Robust int parsing (for ID count)."""
    if x is None:
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    try:
        return int(round(float(s)))  # handles 1, 1.0, "1.0"
    except ValueError:
        return 0

# Infosheet

class OvergroundInfoSheet:
    """
    Expected headers:
      - Trials/Events
      - ID                (COUNT of cycles to export for ID)
      - Dynamic           (Yes/No)  (flag)
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
        for c in ["Trials/Events", "Valid GaitCycle", "FP1", "FP2", "FP3", "Dynamic"]:
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

    def id_count(self, trial_name: str) -> int:
        """ID is a COUNT (how many cycles to export)."""
        r = self.row_for_trial(trial_name)
        return max(0, to_int_count(r.get("ID", 0)))

    def dynamic_ok(self, trial_name: str) -> bool:
        """Dynamic is a FLAG (Yes/No). If column missing, assume True."""
        if "Dynamic" not in self.df.columns:
            return True
        r = self.row_for_trial(trial_name)
        return to_yes_flag(r.get("Dynamic", "Yes"))

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


def baseline_correct(
    mot_object: MOT,
    fz_col: str,
    related_cols: list[str],
    plot_debug: bool = False
) -> None:
    if fz_col not in mot_object.data.columns:
        print(f"[DEBUG] {fz_col} not found in .mot file.")
        return

    fy = mot_object.data[fz_col].to_numpy()
    if len(fy) < 3:
        return

    original_fy = fy.copy()
    corrected_df = deepcopy(mot_object.data)

    # Identify valleys (swing phase)
    valley_idx = np.where((fy[1:-1] < fy[:-2]) & (fy[1:-1] < fy[2:]))[0] + 1
    swing_valleys = valley_idx[fy[valley_idx] < 0]

    # Compute baseline
    if len(swing_valleys) == 0:
        print(f"[WARN] No valleys detected for {fz_col}. Using fallback window-based baseline.")
        fallback_idx = np.where(fy < 0)[0]
        baseline = abs(np.median(fy[fallback_idx])) if len(fallback_idx) > 0 else 0.0
    else:
        baseline = abs(np.median(fy[swing_valleys]))

    print(f"[DEBUG] Baseline offset for {fz_col}: {baseline:.3f} N")

    # Apply baseline correction
    corrected_df[fz_col] = corrected_df[fz_col] + baseline
    corrected_fy = corrected_df[fz_col].to_numpy()
    peaks, _ = find_peaks(corrected_fy, height=0.66*np.max(corrected_fy))
    backwards = -1
    for i in range(peaks[0], 0, -1):
        if corrected_fy[i] < 0:
            backwards = i+1
            break
    forward = -1
    for i in range(peaks[-1], corrected_fy.shape[0]):
        if corrected_fy[i] < 0:
            forward = i - 1
            break
    corrected_fy[:backwards] = 0
    corrected_fy[forward:] = 0
    corrected_df[fz_col] = corrected_fy
    # Zero out positive peaks in swing (after baseline correction)
    # if len(swing_valleys) > 0:
    #     for valley in swing_valleys:
    #         window = 100  # frames before and after
    #         start = max(valley - window, 0)
    #         end = min(valley + window, len(corrected_fy))
    #
    #         swing_segment = corrected_fy[start:end]
    #         pos_peaks = np.where(swing_segment > 0)[0]
    #         corrected_fy[start + pos_peaks] = 0.0000
    #
    #     corrected_df[fz_col] = corrected_fy

    # Offset correction for related force/moment columns
    for col in related_cols:
        if col in corrected_df.columns:
            related = mot_object.data[col].to_numpy()
            offset = np.median(related[swing_valleys]) if len(swing_valleys) > 0 else 0
            corrected_df[col] = related - offset if offset > 0 else related + abs(offset)

    mot_object.data = corrected_df

    # Optional 
    if plot_debug:
        time = mot_object.data["time"].to_numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(time, original_fy, label="Before correction", alpha=0.6)
        plt.plot(time, corrected_fy, label="After baseline + swing zeroing", alpha=0.9)
        plt.axhline(0, linestyle="--", color="black", linewidth=0.8)
        plt.title(f"{fz_col} Baseline + Swing-Phase Artifact Removal")
        plt.xlabel("Time (s)")
        plt.ylabel("Vertical GRF (N)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # stp =f"{fz_col} Baseline + Swing-Phase Artifact Removal"
        # plt.savefig("C:\\Users\\tyeu008\\Documents\\test\\{0}".format(stp))
        # print()


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


# HS events
def build_hs_events(mot: MOT,
                    contacts_by_plate: dict[int, list[tuple[int, int]]],
                    fp_map: dict[int, str]) -> dict[str, list[dict]]:
    """
    Returns:
      events["Left"]  = HS events from left-labeled plates
      events["Right"] = HS events from right-labeled plates
      events["Unknown"] = HS events from 'Both' plates (unusable, but can be boundary)
    """
    mot_t = mot.data["time"].to_numpy()
    events = {"Left": [], "Right": [], "Unknown": []}

    for plate, intervals in contacts_by_plate.items():
        role = fp_map.get(plate, "none")  # left/right/unknown/none
        for (hs_idx, to_idx) in intervals:
            hs_time = float(mot_t[int(hs_idx)])
            to_time = float(mot_t[int(to_idx)])
            e = {
                "plate": int(plate),
                "role": role,  # left/right/unknown/none
                "hs_idx": int(hs_idx),
                "to_idx": int(to_idx),
                "hs_time": hs_time,
                "to_time": to_time,
                "unusable_plate": (role == "unknown")
            }
            if role == "left":
                events["Left"].append(e)
            elif role == "right":
                events["Right"].append(e)
            elif role == "unknown":
                events["Unknown"].append(e)

    for k in ("Left", "Right", "Unknown"):
        events[k].sort(key=lambda d: d["hs_time"])
    return events


def peak_force_in_interval(mot: MOT, plate: int, hs_idx: int, to_idx: int) -> float:
    col = f"ground_force{plate}_vy"
    if col not in mot.data.columns:
        return 0.0
    a = int(min(hs_idx, to_idx))
    b = int(max(hs_idx, to_idx))
    vy = mot.data[col].to_numpy()
    a = clamp_int(a, 0, len(vy) - 1)
    b = clamp_int(b, 0, len(vy) - 1)
    if b <= a:
        return 0.0
    return float(np.max(vy[a:b+1]))


def opposite_contact_exists(events: dict[str, list[dict]],
                            side: str,
                            t1: float,
                            t2: float) -> bool:
    """
    Heuristic: require at least one usable HS from the opposite side between t1 and t2.
    Helps avoid pairing a HS with a wrong unknown boundary.
    """
    opp = "Right" if side == "Left" else "Left"
    for e in events.get(opp, []):
        if t1 < e["hs_time"] < t2:
            return True
    return False


def build_candidate_cycles(mot: MOT,
                           events: dict[str, list[dict]],
                           side: str,
                           valid_sides: list[str],
                           min_cycle_s: float = 0.30,
                           max_cycle_s: float = 2.50,
                           require_opposite_between: bool = True) -> list[dict]:
    """
    Candidate cycle = usable HS (side) -> next HS in time among:
      - next usable HS of same side
      - next Unknown HS (Both plate) (as boundary)
    Start HS must be usable. End HS may be usable or unknown.

    Returns list of dict with fields:
      hs1, hs2, side, duration, score
    """
    if side not in valid_sides:
        return []

    usable = events[side]
    unknown = events["Unknown"]

    merged = sorted(usable + unknown, key=lambda d: d["hs_time"])

    candidates = []
    for hs1 in usable:
        # find hs1 position in merged by time+idx
        hs1_time = hs1["hs_time"]

        # find next boundary event after hs1
        next_ev = None
        for ev in merged:
            if ev["hs_time"] > hs1_time + 1e-12:
                next_ev = ev
                break
        if next_ev is None:
            continue

        hs2 = next_ev
        duration = float(hs2["hs_time"] - hs1_time)
        if duration < min_cycle_s or duration > max_cycle_s:
            continue

        if require_opposite_between:
            # only apply if both sides are valid
            if len(valid_sides) == 2:
                if not opposite_contact_exists(events, side, hs1_time, hs2["hs_time"]):
                    continue

        # scoring
        peak = peak_force_in_interval(mot, hs1["plate"], hs1["hs_idx"], hs1["to_idx"])
        hs2_bonus = 1.0 if not hs2.get("unusable_plate", False) else 0.0
        # prefer durations closer to ~1s (soft)
        dur_penalty = abs(duration - 1.0)

        score = (peak * 1e-3) + (2.0 * hs2_bonus) - (0.5 * dur_penalty)

        candidates.append({
            "side": side,
            "hs1": hs1,
            "hs2": hs2,
            "duration": duration,
            "peak_vy": peak,
            "score": score,
            "start_time": hs1_time,
            "end_time": float(hs2["hs_time"]),
        })

    # sort best first
    candidates.sort(key=lambda d: d["score"], reverse=True)
    return candidates


def select_best_cycles(candidates: list[dict],
                       id_count: int,
                       overlap_s: float = 0.20) -> list[dict]:
    """
    Choose top id_count cycles with minimal overlap.
    Two cycles overlap if their time windows overlap more than overlap_s seconds.
    """
    selected = []
    for c in candidates:
        if len(selected) >= id_count:
            break
        ok = True
        for s in selected:
            a1, a2 = c["start_time"], c["end_time"]
            b1, b2 = s["start_time"], s["end_time"]
            overlap = max(0.0, min(a2, b2) - max(a1, b1))
            if overlap > overlap_s:
                ok = False
                break
        if ok:
            selected.append(c)
    # optional: sort selected chronologically for cleaner output
    selected.sort(key=lambda d: d["start_time"])
    return selected

# Segmentation HS->HS using selected cycles


def segment_cycles_for_id(trial: Trial,
                          info: OvergroundInfoSheet,
                          contacts_by_plate: dict[int, list[tuple[int, int]]],
                          save_root: str,
                          pad_s: float = 0.05,
                          min_cycle_s: float = 0.30,
                          max_cycle_s: float = 2.50) -> list[dict]:
    mot = trial.corrected_grf
    trc = trial.trc
    if trc is None:
        raise MissingPathException(
            f"Markers trajectory object (TRC) for trial {trial.name}",
            "No such object given."
        )

    id_count = info.id_count(trial.name)
    valid_sides = info.valid_sides(trial.name)
    fp_map = info.fp_side_map(trial.name)

    print(f"[DEBUG] ID_count: {id_count}")
    print(f"[DEBUG] Valid sides: {valid_sides}")
    print(f"[DEBUG] FP map: {fp_map}")
    print(f"[DEBUG] Contacts per plate: { {p: len(iv) for p, iv in contacts_by_plate.items()} }")

    if id_count <= 0 or len(valid_sides) == 0:
        return []

    mot_t = mot.data["time"].to_numpy()
    trc_t = trc.data["Time"].to_numpy()

    mot_first = int(getattr(mot, "first_frame", 0))
    trc_first = int(getattr(trc, "first_frame", 0))
    mot_n = mot.data.shape[0]
    trc_n = trc.data.shape[0]
    mot_abs_min = mot_first
    mot_abs_max = mot_first + mot_n - 1
    trc_abs_min = trc_first
    trc_abs_max = trc_first + trc_n - 1

    events = build_hs_events(mot, contacts_by_plate, fp_map)

    # Build candidates across allowed sides
    all_candidates = []
    for side in ("Left", "Right"):
        all_candidates.extend(
            build_candidate_cycles(
                mot=mot,
                events=events,
                side=side,
                valid_sides=valid_sides,
                min_cycle_s=min_cycle_s,
                max_cycle_s=max_cycle_s,
                require_opposite_between=True
            )
        )

    if len(all_candidates) == 0:
        print(f"[WARN] No candidate HS->HS cycles found for {trial.name}.")
        return []

    selected = select_best_cycles(all_candidates, id_count=id_count)

    if len(selected) == 0:
        print(f"[WARN] Candidates existed but none selected (overlap/filters) for {trial.name}.")
        return []

    manifest_rows = []
    total_cycles = 0

    for pick in selected:
        side = pick["side"]
        hs1 = pick["hs1"]
        hs2 = pick["hs2"]

        # Start must be usable (it is, by construction)
        if hs1.get("unusable_plate", False):
            continue

        start_time = float(hs1["hs_time"]) - float(pad_s)
        end_time = float(hs2["hs_time"]) + float(pad_s)

        # convert times to indices
        mot_start_idx = clamp_int(nearest_index(mot_t, start_time), 0, mot_n - 1)
        mot_end_idx = clamp_int(nearest_index(mot_t, end_time), 0, mot_n - 1)
        trc_start_idx = clamp_int(nearest_index(trc_t, start_time), 0, trc_n - 1)
        trc_end_idx = clamp_int(nearest_index(trc_t, end_time), 0, trc_n - 1)

        if mot_end_idx <= mot_start_idx or trc_end_idx <= trc_start_idx:
            continue

        # absolute frames
        mot_start_frame = clamp_int(mot_first + int(mot_start_idx), mot_abs_min, mot_abs_max)
        mot_end_frame = clamp_int(mot_first + int(mot_end_idx), mot_abs_min, mot_abs_max)
        trc_start_frame = clamp_int(trc_first + int(trc_start_idx), trc_abs_min, trc_abs_max)
        trc_end_frame = clamp_int(trc_first + int(trc_end_idx), trc_abs_min, trc_abs_max)

        if mot_end_frame <= mot_start_frame or trc_end_frame <= trc_start_frame:
            continue

        try:
            grf_seg = mot.sample(mot_start_frame, mot_end_frame)
            trc_seg = trc.sample(trc_start_frame, trc_end_frame)
        except Exception as e:
            print(f"[WARN] Skipping selected cycle sampling failed: {trial.name} {side}: {repr(e)}")
            continue

        # add cycle object
        cycle_num = len(trial.gait_cycles[side]) + 1
        cycle = GaitCycle(side=side, number=cycle_num)
        cycle.forceplate_num = int(hs1["plate"])  # plate of the START HS (usable)

        cycle.add_grf(grf_object=grf_seg)
        cycle.add_trc(trc_object=trc_seg)
        trial.gait_cycles[side].append(cycle)

        # save
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
            "id_count_requested": id_count,
            "start_plate": int(hs1["plate"]),
            "end_plate": int(hs2["plate"]),
            "end_plate_unusable": bool(hs2.get("unusable_plate", False)),
            "hs1_time": float(hs1["hs_time"]),
            "hs2_time": float(hs2["hs_time"]),
            "start_time": float(mot_t[mot_start_idx]),
            "end_time": float(mot_t[mot_end_idx]),
            "duration_s": float(pick["duration"]),
            "peak_vy_start": float(pick["peak_vy"]),
            "score": float(pick["score"]),
            "grf_path": cycle.grf.filepath,
            "trc_path": cycle.trc.filepath,
        })

        total_cycles += 1

    if total_cycles == 0:
        print(f"[WARN] No cycles segmented after selection for {trial.name}.")
    else:
        print(f"[INFO] Segmented {total_cycles}/{id_count} cycle(s) for ID in {trial.name}.")

    return manifest_rows



def load_trial_objects(trial_name: str, grf_path: str, trc_path: str) -> Trial:
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
        baseline_correct(corrected, vy, [f"ground_force{plate}_vx", f"ground_force{plate}_vz"], plot_debug=True)

    trial.add_corrected_grf(corrected_grf=corrected)

    safe_mkdir(corrected_out)
    corrected.save(corrected_out)

    # Dynamic flag
    if not info.dynamic_ok(trial.name):
        print(f"[INFO] Skipping segmentation for {trial.name}: Dynamic != Yes.")
        return []

    # ID is COUNT
    id_count = info.id_count(trial.name)
    if id_count <= 0:
        print(f"[INFO] Skipping segmentation for {trial.name}: ID_count <= 0.")
        return []

    contacts = detect_overground_contacts(corrected, fs, threshold=threshold)

    safe_mkdir(segmented_out)
    return segment_cycles_for_id(
        trial=trial,
        info=info,
        contacts_by_plate=contacts,
        save_root=segmented_out,
        pad_s=0.05,
        min_cycle_s=0.30,
        max_cycle_s=2.50
    )


def from_app(args):
    DATA_ROOT = args['directory']
    PARTICIPANT = args['participant_id']
    INFO_CSV_NAME = args['csv_name']
    CONTACT_THRESHOLD_N = args['threshold']
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

    print("\n[Done] GRF correction + ID-cycle segmentation completed.")
    print(f"[Done] Manifest written: {manifest_path}")
    print(f"[Done] Total segmented cycles: {len(df)}")

# MAIN

def main():
    DATA_ROOT = r"C:\Users\tyeu008\Documents\example"
    PARTICIPANT = "PLB_02"
    INFO_CSV_NAME = "Trials_PLB_02.csv"
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

    print("\n[Done] GRF correction + ID-cycle segmentation completed.")
    print(f"[Done] Manifest written: {manifest_path}")
    print(f"[Done] Total segmented cycles: {len(df)}")


if __name__ == "__main__":
    main()

    # test