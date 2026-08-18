"""
data_postprocessing_V2.py
=========================
Version 2 of the TreadMetrix data post-processing pipeline.

WHAT IS DIFFERENT FROM data_postprocessing.py
----------------------------------------------
The ONLY change from V1 is in how gait events (heel strikes and toe-offs)
are detected. All other functions (filter_grf, baseline_correct_debug,
zero_swing_phase, plot_grf_details, segment_at_heel_strikes, process) are
identical to data_postprocessing.py in name, signature, and pipeline order.

WHY A V2?
---------
Some subjects exhibit pronounced weight-shifting during mid-stance, causing
the vertical GRF signal to dip mid-stance without ever reaching zero force.
This creates multiple humps in a single stance phase:

      Normal subject:          Weight-shifting subject (e.g. P02):
      ------                   ----    ----
      |    |                   |  |    |  |
    --      --               --    ----    --
       heel   toe              heel  dip  toe

In V1, the detect_toe_offs and detect_heel_strikes functions use
scipy.signal.find_peaks to find the loading-response and push-off peaks,
then search forward/backward for the threshold crossing. When there are
multiple humps per stance, find_peaks returns extra peaks inside stance,
causing false heel strikes to be detected in mid-stance.

THE V2 SOLUTION - STANCE PHASE BOUNDARY DETECTION
--------------------------------------------------
Instead of searching for force peaks, V2 uses detect_stance_phases() to
identify contiguous windows where Fy > threshold as a single stance phase.
From those windows:

  * Heel strike = first frame where Fy crosses above threshold (window onset)
  * Toe-off     = first frame where Fy crosses below threshold (window offset)

Weight-shifting mid-stance dips are ignored as long as Fy does not fall
below the threshold. The number of humps within stance is irrelevant.

PIPELINE ORDER (identical to V1)
---------------------------------
  1. filter_grf                 - 4th-order Butterworth LP filter (15 Hz)
  2. baseline_correct_debug     - dynamic swing-valley baseline subtraction
  3. detect_toe_offs_V2         - NEW: stance-boundary toe-off detection
  4. detect_heel_strikes_V2     - NEW: stance-boundary heel-strike detection
  5. zero_swing_phase           - zero all GRF channels during swing
  6. plot_grf_details           - plot + interactive span selector
  7. segment_at_heel_strikes    - slice MOT & TRC at heel-strike pairs
"""

from __future__ import annotations

from matplotlib.widgets import SpanSelector
from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
from resources.custom_exceptions import MissingPathException
import os
from copy import deepcopy
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from resources.trial_class import Trial, GaitCycle


# ---------------------------------------------------------------------------
# Global span-selector state (same as V1)
# ---------------------------------------------------------------------------
selected_start = -1
selected_end = -1


def reinitialize_inputs():
    global selected_start, selected_end
    selected_start = -1
    selected_end = -1


# ===========================================================================
# STEP 1 - GRF Low-Pass Filter  (IDENTICAL TO V1)
# ===========================================================================

def filter_grf(mot: MOT, fs: float, cutoff: float = 15.0, order: int = 4) -> None:
    """Filters data of a MOT object with a 4th-order Butterworth low-pass filter.

    Args:
        mot: MOT object whose data is to be filtered.
        fs: sampling frequency in Hz.
        cutoff: cutoff frequency in Hz (default 15.0 Hz).
        order: filter order (default 4).
    """
    if len(mot.data) <= 3:
        return
    nyq = 0.5 * fs
    normal_cutoff = min(0.99, cutoff / nyq)
    b, a = butter(order, normal_cutoff, btype='low', output='ba')
    padlen = min(15, len(mot.data) - 1)
    filtered_df = deepcopy(mot.data)
    for col in mot.data.columns.tolist():
        if col.lower() == 'time':
            continue
        filtered_df[col] = filtfilt(b, a, mot.data[col], padlen=padlen)
    mot.data = filtered_df


# ===========================================================================
# STEP 2 - Baseline Correction  (IDENTICAL TO V1)
# ===========================================================================

def baseline_correct_debug(mot_object: MOT, fz_col: str, related_cols: list[str],
                           output_path: str = None, show: bool = False) -> None:
    """Corrects the baseline of one of the columns of the mot data.

    Identifies swing-phase valleys (low-force periods) and subtracts a
    linearly interpolated baseline from the force signal and its related
    columns.  Saves an optional diagnostic plot.

    Args:
        mot_object: data to process.
        fz_col: name of the vertical force column to correct.
        related_cols: other columns sharing the same offset (Fx, Fz).
        output_path: output path for plot save. Optional. If None, not saved.
        show: whether to show the figure when method is called.
    """
    fy = mot_object.data[fz_col]
    corrected_df = deepcopy(mot_object.data)

    # Invert the signal because find_peaks finds local maxima.
    valley_indices, _ = find_peaks(-fy)

    print(f"\nCorrecting {fz_col}")

    if len(valley_indices) == 0:
        print("No valleys found. Skipping correction.")
        return

    # Calculate sampling frequency (fs) from time column
    time_arr = mot_object.data['time'].values
    fs = int(round(1.0 / (time_arr[1] - time_arr[0])))

    # Find all valleys with minimum distance between them
    valley_indices, _ = find_peaks(-fy, distance=fs // 4)

    print(f"\nCorrecting {fz_col}")

    if len(valley_indices) == 0:
        print("No valleys found. Skipping correction.")
        return

    # Keep only true swing valleys (lower 30 % of signal range)
    signal_range = np.max(fy) - np.min(fy)
    upper_threshold = np.min(fy) + 0.3 * signal_range
    swing_valleys = valley_indices[fy[valley_indices] <= upper_threshold]

    print(f"Number of swing valleys found: {len(swing_valleys)}")

    if len(swing_valleys) < 2:
        print("Not enough swing valleys for interpolation. Using median.")
        baseline_array = np.full(len(fy),
                                 np.median(fy[swing_valleys]) if len(swing_valleys) > 0 else 0)
    else:
        valley_times = swing_valleys
        valley_values = fy[swing_valleys]
        interp_func = interp1d(valley_times, valley_values, kind='linear',
                               bounds_error=False,
                               fill_value=(valley_values.iloc[0], valley_values.iloc[-1]))
        baseline_array = interp_func(np.arange(len(fy)))

    corrected_df[fz_col] = fy - baseline_array

    for col in related_cols:
        related = mot_object.data[col]
        if len(swing_valleys) < 2:
            offset_array = np.full(len(related),
                                   np.median(related[swing_valleys]) if len(swing_valleys) > 0 else 0)
        else:
            rel_values = related[swing_valleys]
            interp_func_rel = interp1d(valley_times, rel_values, kind='linear',
                                       bounds_error=False,
                                       fill_value=(rel_values.iloc[0], rel_values.iloc[-1]))
            offset_array = interp_func_rel(np.arange(len(related)))
        corrected_df[col] = related - offset_array
        print(f"Applied dynamic offset for {col}")

    mot_object.data = corrected_df

    if (output_path is not None) or show:
        time_scale = (mot_object.data['time']
                      if 'time' in mot_object.data.columns.tolist()
                      else np.arange(len(fy)))
        plt.figure(figsize=(12, 4))
        plt.plot(time_scale, fy, label='Original', alpha=0.7)
        plt.scatter(time_scale[swing_valleys], fy[swing_valleys],
                    color='red', label='Swing Valleys')
        plt.plot(time_scale, corrected_df[fz_col], label='Corrected', alpha=0.8)
        plt.title(f"{fz_col} Baseline Correction")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()
        plt.grid(True)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            file_name = f"{mot_object.filename.replace('.mot', '')}_baseline_correction_{fz_col}.png"
            plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight')

        if show:
            plt.show()


# ===========================================================================
# STEP 3 & 4 - Gait Event Detection  (NEW IN V2)
# ===========================================================================

def detect_stance_phases(fy: np.ndarray, threshold: float = 25,
                         to_threshold: float = 15,
                         min_stance_frames: int = 50,
                         min_swing_frames: int = 20) -> list[tuple[int, int]]:
    """Detect contiguous stance windows from a vertical GRF signal with hysteresis.

    A stance window begins when Fy crosses above `threshold` (default 25 N for Heel Strike)
    and ends when Fy drops below `to_threshold` (default 15 N for Toe-Off).
    Short artefact bursts (< min_stance_frames) are discarded.
    Short swing gaps (< min_swing_frames) are merged into the surrounding
    stance so that a brief mid-stance dip that just touches the threshold
    is not split into two separate stance phases.

    KEY ADVANTAGE OVER V1 PEAK-FINDING:
    Multiple humps within a single stance (caused by mid-stance weight
    shifting) are automatically merged into one stance window.  No false
    heel strikes are generated in mid-stance.

    Args:
        fy:                 1-D array of vertical GRF (N).
        threshold:          Force threshold (N) for Heel Strike onset (default 25 N).
        to_threshold:       Force threshold (N) for Toe-Off release (default 15 N).
        min_stance_frames:  Minimum number of consecutive frames above
                            threshold to count as a real stance phase.
                            At 1000 Hz, default 50 = 50 ms minimum contact.
        min_swing_frames:   Minimum number of consecutive frames below
                            threshold to separate two distinct stances.
                            At 1000 Hz, default 20 = 20 ms minimum swing gap.
                            Gaps shorter than this are merged into one stance.

    Returns:
        List of (onset, offset) index pairs, one per detected stance phase.
        onset  = first frame where Fy crosses above threshold (Heel Strike).
        offset = first frame where Fy crosses below to_threshold after onset (Toe-Off).
    """
    fy = np.asarray(fy)
    base_thresh = min(threshold, to_threshold) if to_threshold is not None else threshold
    above = (fy > base_thresh).astype(int)

    # ---- merge short swing gaps ----------------------------------------
    # Find starts and ends of below-threshold segments
    diff = np.diff(np.concatenate(([0], above, [0])))
    swing_starts = np.where(diff == -1)[0]   # transitions from above -> below
    swing_ends   = np.where(diff ==  1)[0]   # transitions from below -> above
    for ss, se in zip(swing_starts, swing_ends):
        gap = se - ss
        if gap < min_swing_frames:
            above[ss:se] = 1   # bridge the short gap — treat as continuous stance

    # ---- collect stance windows -----------------------------------------
    diff2 = np.diff(np.concatenate(([0], above, [0])))
    onsets  = np.where(diff2 ==  1)[0]
    offsets = np.where(diff2 == -1)[0]

    stances = []
    for on, off in zip(onsets, offsets):
        if (off - on) >= min_stance_frames:
            segment = fy[on:off]
            if np.max(segment) >= threshold:
                hs_cross = np.where(segment >= threshold)[0]
                actual_on = on + (hs_cross[0] if len(hs_cross) > 0 else 0)
                stances.append((int(actual_on), int(off)))

    return stances


def detect_toe_offs_V2(mot: MOT, threshold: float = 15,
                       hs_threshold: float = 25,
                       min_stance_frames: int = 50,
                       min_swing_frames: int = 20) -> dict[str, list[int]]:
    """Detect toe-offs using stance-phase boundary detection (V2).

    Toe-off = the first frame where Fy crosses below threshold (default 15 N) after
    a complete stance phase.  Mid-stance weight-shifting dips are merged
    into a single stance window and do not generate spurious toe-offs.

    Args:
        mot:               MOT object (post-filter, post-baseline-correct).
        threshold:         Force threshold in N for toe-off (default 15 N).
        hs_threshold:      Force threshold in N for stance onset (default 25 N).
        min_stance_frames: Minimum stance duration in frames.
        min_swing_frames:  Minimum swing gap to separate two stances.

    Returns:
        {'R': [...], 'L': [...]} - frame indices of toe-offs.
    """
    toe_offs = {'R': [], 'L': []}
    max_idx = len(mot.data) - 1

    if 'ground_force5_vy' in mot.data.columns:
        rzf = mot.data['ground_force5_vy'].values
        stances = detect_stance_phases(rzf, threshold=hs_threshold, to_threshold=threshold,
                                       min_stance_frames=min_stance_frames, min_swing_frames=min_swing_frames)
        toe_offs['R'] = [min(off, max_idx) for (on, off) in stances if on > 0]

    if 'ground_force4_vy' in mot.data.columns:
        lzf = mot.data['ground_force4_vy'].values
        stances = detect_stance_phases(lzf, threshold=hs_threshold, to_threshold=threshold,
                                       min_stance_frames=min_stance_frames, min_swing_frames=min_swing_frames)
        toe_offs['L'] = [min(off, max_idx) for (on, off) in stances if on > 0]

    return toe_offs


def detect_heel_strikes_V2(mot: MOT, threshold: float = 25,
                           to_threshold: float = 15,
                           min_stance_frames: int = 50,
                           min_swing_frames: int = 20) -> dict[str, list[int]]:
    """Detect heel strikes using stance-phase boundary detection (V2).

    Heel strike = the first frame where Fy crosses above threshold (default 25 N) at
    the onset of a stance phase.  Multiple humps within a single stance
    phase (caused by mid-stance weight shifting) are treated as one
    continuous stance and do not produce spurious heel strikes.

    Args:
        mot:               MOT object (post-filter, post-baseline-correct).
        threshold:         Force threshold in N for heel strike (default 25 N).
        to_threshold:      Force threshold in N for toe-off (default 15 N).
        min_stance_frames: Minimum stance duration in frames.
        min_swing_frames:  Minimum swing gap to separate two stances.

    Returns:
        {'R': [...], 'L': [...]} - frame indices of heel strikes.
    """
    heel_contacts = {'R': [], 'L': []}
    max_idx = len(mot.data) - 1

    if 'ground_force5_vy' in mot.data.columns:
        rzf = mot.data['ground_force5_vy'].values
        stances = detect_stance_phases(rzf, threshold=threshold, to_threshold=to_threshold,
                                       min_stance_frames=min_stance_frames, min_swing_frames=min_swing_frames)
        heel_contacts['R'] = [min(on, max_idx) for (on, _) in stances if on > 0]

    if 'ground_force4_vy' in mot.data.columns:
        lzf = mot.data['ground_force4_vy'].values
        stances = detect_stance_phases(lzf, threshold=threshold, to_threshold=to_threshold,
                                       min_stance_frames=min_stance_frames, min_swing_frames=min_swing_frames)
        heel_contacts['L'] = [min(on, max_idx) for (on, _) in stances if on > 0]

    return heel_contacts


# ===========================================================================
# STEP 5 - Zero Swing Phase  (IDENTICAL TO V1)
# ===========================================================================

def zero_swing_phase(mot_df: MOT, toe_offs: dict[str, list[int]],
                     heel_strikes: dict[str, list[int]], side: str) -> None:
    """Sets GRF and related columns to zero between toe-off and next heel strike.

    Args:
        mot_df:       MOT object to modify in-place.
        toe_offs:     toe-off moments, listed by side.
        heel_strikes: heel-strike moments, listed by side.
        side:         'R'/'r'/'right' or 'L'/'l'/'left'.

    Raises:
        ValueError: if side is not a recognised string.
    """
    df_corrected = deepcopy(mot_df.data)

    side = side.lower()

    if side in ('r', 'right'):
        to_list = toe_offs['R']
        hs_list = heel_strikes['R']
        cols_to_zero = ['ground_force5_vx', 'ground_force5_vy', 'ground_force5_vz',
                        'ground_force5_px', 'ground_force5_py', 'ground_force5_pz',
                        'ground_torque5_x', 'ground_torque5_y', 'ground_torque5_z']
    elif side in ('l', 'left'):
        to_list = toe_offs['L']
        hs_list = heel_strikes['L']
        cols_to_zero = ['ground_force4_vx', 'ground_force4_vy', 'ground_force4_vz',
                        'ground_force4_px', 'ground_force4_py', 'ground_force4_pz',
                        'ground_torque4_x', 'ground_torque4_y', 'ground_torque4_z']
    else:
        raise ValueError("Side must be 'R' or 'L'")

    for toe_idx in to_list:
        hs_after_toe = [hs for hs in hs_list if hs > toe_idx]
        if hs_after_toe:
            heel_idx = hs_after_toe[0]
            for col in cols_to_zero:
                if col in df_corrected.columns.tolist():
                    df_corrected.loc[toe_idx:heel_idx - 1, col] = 0

    mot_df.data = df_corrected


# ===========================================================================
# STEP 6 - GRF Plot with Interactive Span Selector  (IDENTICAL TO V1)
# ===========================================================================

def plot_grf_details(mot: MOT, heel_strikes: dict[str, list[int]],
                     toe_offs: dict[str, list[int]], output: str,
                     show: bool = True) -> None:
    """Save plot of vertical forces with toe-offs and heel strikes.

    Args:
        mot:          MOT object of the data.
        heel_strikes: heel-strike frames by side.
        toe_offs:     toe-off frames by side.
        output:       output directory for the PNG.
        show:         whether to display the interactive plot window.
    """
    global selected_start, selected_end
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

    ax1.set_title(f"Vertical GRFs with Toe-Offs and Heel Strikes: {mot.filename}")
    ax2.set_title('Data to process')
    ax1.set_xlabel("Time [s]")
    ax2.set_xlabel("Time [s]")
    ax1.set_ylabel('Force [N]')
    ax2.set_ylabel('Force [N]')
    ax1.grid(True)
    ax2.grid(True)
    fig.tight_layout()

    os.makedirs(output, exist_ok=True)

    time_scale = (mot.data['time'].values
                  if 'time' in mot.data.columns.tolist()
                  else np.arange(mot.data.shape[0]))
    right_fy = mot.data['ground_force5_vy'].values
    left_fy  = mot.data['ground_force4_vy'].values

    selected_start = 0
    selected_end   = len(time_scale) - 1

    ax1.plot(time_scale, right_fy, label='Right Fy', alpha=0.7, color='orange')
    ax1.plot(time_scale, left_fy,  label='Left Fy',  alpha=0.7, color='green')

    # Toe-offs
    ax1.scatter([time_scale[i] for i in toe_offs['R']],
                [right_fy[i]  for i in toe_offs['R']],
                color='darkorange', marker='x', label='Right Toe-Offs')
    ax1.scatter([time_scale[i] for i in toe_offs['L']],
                [left_fy[i]   for i in toe_offs['L']],
                color='darkgreen',  marker='x', label='Left Toe-Offs')

    # Heel strikes
    ax1.scatter([time_scale[i] for i in heel_strikes['R']],
                [right_fy[i]  for i in heel_strikes['R']],
                color='darkorange', marker='o', label='Right Heel Strikes')
    ax1.scatter([time_scale[i] for i in heel_strikes['L']],
                [left_fy[i]   for i in heel_strikes['L']],
                color='darkgreen',  marker='o', label='Left Heel Strikes')

    def onselect(xmin, xmax):
        global selected_start, selected_end
        indmin, indmax = np.searchsorted(time_scale, (xmin, xmax))
        indmax = min(len(time_scale) - 1, indmax)
        x_time  = list(time_scale[indmin:indmax])
        y_right = right_fy[indmin:indmax]
        y_left  = left_fy[indmin:indmax]
        if len(x_time) >= 2:
            ax2.plot(x_time, y_right, label='Right Fy', alpha=0.7, color='orange')
            ax2.plot(x_time, y_left,  label='Left Fy',  alpha=0.7, color='green')
            ax2.set_xlim(x_time[0], x_time[-1])
            ax2.set_ylim(min(y_right.min(), y_left.min()),
                         max(y_right.max(), y_left.max()))
            fig.canvas.draw_idle()
            selected_start = indmin
            selected_end   = indmax

    span = SpanSelector(
        ax1, onselect, "horizontal",
        useblit=True,
        props=dict(alpha=0.5, facecolor="tab:grey"),
        interactive=True,
        drag_from_anywhere=True,
    )

    fig.legend()
    plt.savefig(os.path.join(output,
                             f"{mot.filename.replace('.mot', '')}"
                             f"_vertical_grfs_with_toeoffs_heelstrikes.png"),
                bbox_inches='tight')
    if show:
        plt.show()


# ===========================================================================
# STEP 7 - Segment at Heel Strikes  (IDENTICAL TO V1)
# ===========================================================================

def segment_at_heel_strikes(trial: Trial, heel_strike_moments: dict[str, list[int]],
                            mot_frame_rate: float = None, save: str = None) -> None:
    """Segment MOT (and matching TRC) objects at heel strikes.

    Args:
        trial:                Trial object for the trial being processed.
        heel_strike_moments:  dictionary of heel strikes by side ('R'/'L').
        mot_frame_rate:       MOT frame rate in Hz. Optional (computed if None).
        save:                 Directory to write segmented files. Optional.
    """
    mot = trial.corrected_grf
    trc = trial.trc

    # Segment MOT
    right_mots = mot.segment(heel_strike_moments['R'], True)[1:-1]
    left_mots  = mot.segment(heel_strike_moments['L'], True)[1:-1]

    for i, m in enumerate(right_mots):
        m.name = f"{trial.name}_Right_cycle{i}"
        m.filename = f"{trial.name}_Right_cycle{i}.mot"

    for i, m in enumerate(left_mots):
        m.name = f"{trial.name}_Left_cycle{i}"
        m.filename = f"{trial.name}_Left_cycle{i}.mot"

    # Apply secondary 6.0 Hz low-pass filter on segmented GRF forces prior to ID calculation
    if mot_frame_rate is None:
        mot_frame_rate = 1 / np.mean(np.diff(mot.data['time']))
    for m in right_mots:
        filter_grf(m, mot_frame_rate, cutoff=6.0, order=4)
    for m in left_mots:
        filter_grf(m, mot_frame_rate, cutoff=6.0, order=4)

    if save is not None:
        right_path = os.path.join(save, "Right")
        left_path  = os.path.join(save, "Left")
        MOT.save_multiple(right_mots, right_path)
        MOT.save_multiple(left_mots,  left_path)
        trial.gait_cycles["Right"] = GaitCycle.to_gait_cycles(
            side="Right", grfs=right_mots, grf_path=right_path)
        trial.gait_cycles["Left"]  = GaitCycle.to_gait_cycles(
            side="Left",  grfs=left_mots,  grf_path=left_path)
    else:
        trial.gait_cycles["Right"] = GaitCycle.to_gait_cycles(
            side="Right", grfs=right_mots)
        trial.gait_cycles["Left"]  = GaitCycle.to_gait_cycles(
            side="Left",  grfs=left_mots)

    # Segment TRC
    if trc is not None:
        trc_rate = trc.metadata.camera_rate
        if mot_frame_rate is None:
            mot_frame_rate = 1 / np.mean(np.diff(mot.data['time']))
        if trc_rate is None:
            trc_rate = 1 / np.mean(np.diff(trc.data['Time']))
        rate_conversion = trc_rate / mot_frame_rate

        trc_heel_strike_moments = {side: [] for side in heel_strike_moments}
        ff = trc.first_frame
        lf = ff + len(trc.data) - 1

        for side in heel_strike_moments:
            for i in range(len(heel_strike_moments[side])):
                frame_num = int(round(heel_strike_moments[side][i] * rate_conversion)) + ff
                frame_num = max(ff, min(lf, frame_num))
                trc_heel_strike_moments[side].append(frame_num)
            trc_heel_strike_moments[side] = sorted(list(set(trc_heel_strike_moments[side])))

        right_trcs = trc.segment(trc_heel_strike_moments['R'], True)[1:-1] if len(trc_heel_strike_moments['R']) >= 2 else []
        left_trcs  = trc.segment(trc_heel_strike_moments['L'], True)[1:-1] if len(trc_heel_strike_moments['L']) >= 2 else []


        for i, t in enumerate(right_trcs):
            t.name = f"{trial.name}_Right_cycle{i}"
            t.filename = f"{trial.name}_Right_cycle{i}.trc"

        for i, t in enumerate(left_trcs):
            t.name = f"{trial.name}_Left_cycle{i}"
            t.filename = f"{trial.name}_Left_cycle{i}.trc"

        if save is not None:
            right_path = os.path.join(save, "Right")
            left_path  = os.path.join(save, "Left")
            TRC.save_multiple(right_trcs, right_path)
            TRC.save_multiple(left_trcs,  left_path)
            GaitCycle.add_to_gait_cycles(trial.gait_cycles["Right"],
                                         trcs=right_trcs, trc_path=right_path)
            GaitCycle.add_to_gait_cycles(trial.gait_cycles["Left"],
                                         trcs=left_trcs,  trc_path=left_path)
        else:
            GaitCycle.add_to_gait_cycles(trial.gait_cycles["Right"], trcs=right_trcs)
            GaitCycle.add_to_gait_cycles(trial.gait_cycles["Left"],  trcs=left_trcs)


# ===========================================================================
# STANDALONE / LEGACY ENTRY POINT  (not called by full_pipeline.py)
# ===========================================================================

def process(trial: Trial, save_plot_path: str, save_segmented_path: str = None,
            show: bool = True, save_optionals: bool = False,
            hs_threshold: float = 25.0,
            to_threshold: float = 15.0,
            min_stance_frames: int = 50,
            min_swing_frames: int = 20) -> None:
    """Standalone pipeline to process the raw data of a trial (Version 2).

    NOTE: This function is kept as a legacy/standalone fallback only.
    The main pipeline (full_pipeline.py) no longer calls this function.
    Instead, full_pipeline.py calls each step explicitly in this order:
      1. Treadmill offset correction  (TreadmillOffsetCorrector)
      2. filter_grf                   (15 Hz LP filter)
      3. detect_toe_offs_V2 / detect_heel_strikes_V2
      4. Interactive Gait Event GUI   (run_interactive_selector)
      5. baseline_correct_debug       (Right then Left)
      6. zero_swing_phase
      7. segment_at_heel_strikes

    This standalone version uses the old step order:
      1. filter_grf
      2. baseline_correct_debug (Right then Left)
      3. detect_toe_offs_V2
      4. detect_heel_strikes_V2
      5. zero_swing_phase
      6. plot_grf_details
      7. segment_at_heel_strikes

    Additional parameters vs V1:
        hs_threshold:       Force threshold (N) for heel strike boundary (default 25 N).
        to_threshold:       Force threshold (N) for toe-off boundary (default 15 N).
        min_stance_frames:  Minimum frames above threshold to count as
                            a real stance.  Increase to filter out brief
                            artefact contacts.  Default = 50 frames = 50 ms
                            at 1000 Hz.
        min_swing_frames:   Minimum frames below threshold to treat as a
                            real swing gap between two stances.  Decrease
                            to allow the mid-stance dip to bridge across.
                            Default = 20 frames = 20 ms at 1000 Hz.

    Args:
        trial:               Trial object to process.
        save_plot_path:      Directory for plots and corrected GRF MOT file.
        save_segmented_path: Directory for segmented MOT/TRC files.
        show:                Whether to display the interactive plot window.
        save_optionals:      Whether to save plots and corrected GRF.
        hs_threshold:        Threshold force (N) for heel strike (default 25 N).
        to_threshold:        Threshold force (N) for toe-off (default 15 N).
        min_stance_frames:   Min frames in stance to count as a real contact.
        min_swing_frames:    Min frames in swing to split two stance phases.
    """
    global selected_start, selected_end
    frame_rate = 1 / np.mean(np.diff(trial.grf.data['time']))
    print(f"\n[V2] Processing: {trial.grf.filename} "
          f"with sampling frequency: {frame_rate:.2f} Hz.")

    # ------------------------------------------------------------------
    # Step 1 - Filter GRF
    # ------------------------------------------------------------------
    corrected_grf = trial.grf.copy()
    corrected_grf.rename(name=trial.name, filename=trial.name + ".mot")
    filter_grf(corrected_grf, frame_rate)

    # ------------------------------------------------------------------
    # Step 2 - Baseline Correction
    # ------------------------------------------------------------------
    baseline_correct_debug(corrected_grf,
                           'ground_force5_vy',
                           ['ground_force5_vx', 'ground_force5_vz'],
                           output_path=save_plot_path if save_optionals else None,
                           show=show)
    baseline_correct_debug(corrected_grf,
                           'ground_force4_vy',
                           ['ground_force4_vx', 'ground_force4_vz'],
                           output_path=save_plot_path if save_optionals else None,
                           show=show)

    # ------------------------------------------------------------------
    # Step 3 - Detect Toe-Offs  (V2: stance-boundary detection)
    # ------------------------------------------------------------------
    toe_off_moments = detect_toe_offs_V2(
        corrected_grf, threshold=to_threshold, hs_threshold=hs_threshold,
        min_stance_frames=min_stance_frames,
        min_swing_frames=min_swing_frames)
    print(f"  Right toe-offs  detected: {len(toe_off_moments['R'])}")
    print(f"  Left  toe-offs  detected: {len(toe_off_moments['L'])}")

    # ------------------------------------------------------------------
    # Step 4 - Detect Heel Strikes  (V2: stance-boundary detection)
    # ------------------------------------------------------------------
    heel_strike_moments = detect_heel_strikes_V2(
        corrected_grf, threshold=hs_threshold, to_threshold=to_threshold,
        min_stance_frames=min_stance_frames,
        min_swing_frames=min_swing_frames)
    print(f"  Right heel strikes detected: {len(heel_strike_moments['R'])}")
    print(f"  Left  heel strikes detected: {len(heel_strike_moments['L'])}")

    # ------------------------------------------------------------------
    # Step 5 - Zero Swing Phase
    # ------------------------------------------------------------------
    zero_swing_phase(corrected_grf, toe_off_moments, heel_strike_moments, 'right')
    zero_swing_phase(corrected_grf, toe_off_moments, heel_strike_moments, 'left')

    # ------------------------------------------------------------------
    # Step 6 - Plot GRF Details with Interactive Span Selector
    # ------------------------------------------------------------------
    plot_grf_details(corrected_grf, heel_strike_moments, toe_off_moments,
                     save_plot_path, show=show)

    corrected_grf = corrected_grf.sample(int(selected_start), int(selected_end))
    corrected_grf.rename(name=trial.name, filename=trial.name + ".mot")
    for side in ["L", "R"]:
        heel_strike_moments[side] = [
            strike for strike in heel_strike_moments[side]
            if (selected_start <= strike <= selected_end)
        ]

    if save_optionals:
        corrected_grf.save(save_plot_path)
        trial.add_corrected_grf(
            corrected_grf=corrected_grf,
            path_to_corrected_grf=os.path.join(save_plot_path, corrected_grf.filename))
    else:
        trial.add_corrected_grf(corrected_grf=corrected_grf)

    # ------------------------------------------------------------------
    # Step 7 - Segment at Heel Strikes
    # ------------------------------------------------------------------
    if trial.trc is None:
        raise MissingPathException(
            f"Markers trajectory object (TRC) for trial {trial.name}",
            "No such object given.")
    segment_at_heel_strikes(trial, heel_strike_moments, save=save_segmented_path)

    reinitialize_inputs()
