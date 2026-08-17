import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re
import resources.paths.paths_access as local
from resources.custom_exceptions import MissingPathException
from resources.trial_class import Trial
import osim_gestion as osim
import data_postprocessing as post_processing_v1
import data_postprocessing_V2 as post_processing_v2
from ik_computing import process as compute_ik
from id_computing import process as compute_id
from joint_power_computing import process as compute_jp
from offset_corrector import TreadmillOffsetCorrector
from interactive_gait_selector import run_interactive_selector
import numpy as np


def make_speed_label(speed, slope):
    """
    Build a clean folder name from speed and slope.
    e.g. speed=2.0, slope=0  ->  'Speed2slope0'
         speed=1.5, slope=5  ->  'Speed1_5slope5'
    """
    speed_str = str(speed).replace('.', '_').rstrip('_0').rstrip('_') or '0'
    slope_str = str(int(slope)) if slope == int(slope) else str(slope).replace('.', '_')
    return f"Speed{speed_str}slope{slope_str}"


def get_cycle_speed_label(cycle, selections, default_label='SpeedUnknown'):
    """
    Determine the speed/slope label for a gait cycle by checking which
    selected time window its start time falls into.
    Uses GaitCycle.get_time_frame() which returns (start_time, end_time).
    """
    time_frame = cycle.get_time_frame()
    if time_frame is None:
        return default_label
    cycle_start = float(time_frame[0])

    for sel in selections:
        if sel['tmin'] <= cycle_start <= sel['tmax']:
            return make_speed_label(sel['speed'], sel['slope'])

    return default_label


if __name__ == "__main__":

    # quick setup for debug
    if local.call_quick_setup():
        save = False
        show = False
        use_offset_corrector = False
        event_detection_mode = 'hybrid'

    else:
        # update local paths:
        local.main_gui()
        osim.main()
        # ask user's preference
        save = local.call_should_save()
        show = local.call_should_show()
        use_offset_corrector = local.call_should_use_offset_corrector()
        event_detection_mode = local.call_event_detection_mode()

    # loads files into Trial objects:
    trials = {}

    try:
        mot_files = local.get_raw_mot_path()
        trc_files = local.get_raw_trc_path()
    except MissingPathException:
        try:
            directory = local.get_raw_directory()
            mot_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".mot")]
            trc_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".trc")]
        except MissingPathException:
            mot_files = []
            trc_files = []
            print("No valid input files or directory selected.")

    mot_files.sort()
    trc_files.sort()

    for file in mot_files:
        trial_name = os.path.basename(file).replace('.mot', '')
        if not (show or save):
            trial_name = trial_name.replace('MOT', '')
        trial_name_clean = re.sub(r'\s+', ' ', trial_name)
        try:
            trial = Trial(mot=file)
            # Normalize name to single space to prevent OpenSim double-space path bugs
            trial.name = trial_name_clean
            trial.grf.name = trial_name_clean
            trial.grf.filename = trial_name_clean + ".mot"
            try:
                trc_file = [t for t in trc_files if re.search(re.escape(trial_name) + r"\.trc$", t) is not None][0]
                trial.add_trc(trc_file)
                if trial.trc is not None:
                    trial.trc.name = trial_name_clean
                    trial.trc.filename = trial_name_clean + ".trc"
            except IndexError:
                raise OSError
            trials[trial_name_clean] = trial
        except OSError:
            print(f"Trial could not be loaded from {trial_name_clean}. Skipping.")
            break

    # Initialize offset corrector if enabled
    corrector = TreadmillOffsetCorrector() if use_offset_corrector else None

    # Choose post-processing module based on event detection mode
    # Hybrid (V2) and GRF V2 both use the V2 (stance-boundary) module.
    # Hybrid V1 and GRF V1 use the older peak-detection module.
    is_hybrid = event_detection_mode in ('hybrid', 'hybrid_v1', 'hybrid_v2')
    postproc_version = 'v1' if event_detection_mode in ('grf_v1', 'hybrid_v1') else 'v2'
    post_module = post_processing_v2 if postproc_version == 'v2' else post_processing_v1
    print(f"Using Event Detection: {event_detection_mode.upper()} | "
          f"Post-Processing Module: {'V2 (Stance Boundary)' if postproc_version == 'v2' else 'V1 (Peak Detection)'}")

    # process the trials:
    for name in trials:
        trial = trials[name]

        # Apply interactive offset correction if enabled.
        if use_offset_corrector and corrector is not None:
            print(f"\n--- Correcting Offsets for Trial: {name} ---")
            corrected_df, selections = corrector.interactive_correction(trial.grf.data, trial_name=name)
            trial.grf.data = corrected_df

            print(f"\n  Speed windows selected: {len(selections)}")
            for sel in selections:
                lbl = make_speed_label(sel['speed'], sel['slope'])
                print(f"    {lbl}: {sel['tmin']:.2f}s – {sel['tmax']:.2f}s")
        else:
            print(f"\n--- Skipping Treadmill Offset Corrector for Trial: {name} ---")
            selections = []

        # ------------------------------------------------------------------
        # Step 2: Filter GRF at 15 Hz
        # ------------------------------------------------------------------
        print(f"\n--- Filtering GRF (15 Hz) for Trial: {name} ---")
        corrected_grf = trial.grf.copy()
        corrected_grf.rename(name=trial.name, filename=trial.name + ".mot")
        t_grf = corrected_grf.data['time'].values
        frame_rate = 1 / np.mean(np.diff(t_grf))
        post_module.filter_grf(corrected_grf, frame_rate)
        t_grf = corrected_grf.data['time'].values

        # ------------------------------------------------------------------
        # Step 3: Gait event detection
        #   Hybrid mode  -> Zeni (marker) + GRF (V1 or V2) -> reconcile -> confirmed/suggested
        #   GRF-only     -> GRF detection only, no suggestions
        # ------------------------------------------------------------------
        print(f"\n--- Event Detection ({event_detection_mode.upper()}) for Trial: {name} ---")

        # Always run GRF-based detection (used for both modes)
        if postproc_version == 'v2':
            grf_hs = post_module.detect_heel_strikes_V2(corrected_grf)
            grf_to = post_module.detect_toe_offs_V2(corrected_grf)
        else:
            grf_hs = post_module.detect_heel_strikes(corrected_grf, frame_rate)
            grf_to = post_module.detect_toe_offs(corrected_grf, frame_rate)

        # Convert GRF frame indices to seconds
        grf_hs_t = {s: [float(t_grf[i]) for i in grf_hs[s] if i < len(t_grf)] for s in ['R', 'L']}
        grf_to_t = {s: [float(t_grf[i]) for i in grf_to[s] if i < len(t_grf)] for s in ['R', 'L']}

        if is_hybrid and trial.trc is not None:
            # ---- Hybrid: Zeni marker detection + reconciliation ----
            from marker_event_detection import detect_events_from_markers, reconcile_events
            marker_events = detect_events_from_markers(trial.trc)

            conf_r_hs, sugg_r_hs = reconcile_events(grf_hs_t['R'], marker_events['HS']['R'])
            conf_l_hs, sugg_l_hs = reconcile_events(grf_hs_t['L'], marker_events['HS']['L'])
            conf_r_to, sugg_r_to = reconcile_events(grf_to_t['R'], marker_events['TO']['R'])
            conf_l_to, sugg_l_to = reconcile_events(grf_to_t['L'], marker_events['TO']['L'])

            print(f"  [Hybrid] Right HS: {len(conf_r_hs)} confirmed, {len(sugg_r_hs)} suggested")
            print(f"  [Hybrid] Left  HS: {len(conf_l_hs)} confirmed, {len(sugg_l_hs)} suggested")
            print(f"  [Hybrid] Right TO: {len(conf_r_to)} confirmed, {len(sugg_r_to)} suggested")
            print(f"  [Hybrid] Left  TO: {len(conf_l_to)} confirmed, {len(sugg_l_to)} suggested")

            initial_r_hs, initial_l_hs = conf_r_hs, conf_l_hs
            initial_r_to, initial_l_to = conf_r_to, conf_l_to
            suggested_r_hs, suggested_l_hs = sugg_r_hs, sugg_l_hs
            suggested_r_to, suggested_l_to = sugg_r_to, sugg_l_to

        else:
            # ---- GRF-only: all events are pre-loaded as active, no suggestions ----
            if is_hybrid:
                print("  [Hybrid] No TRC available -- falling back to GRF-only detection.")
            print(f"  Right: {len(grf_hs_t['R'])} HS, {len(grf_to_t['R'])} TO")
            print(f"  Left:  {len(grf_hs_t['L'])} HS, {len(grf_to_t['L'])} TO")

            initial_r_hs, initial_l_hs = grf_hs_t['R'], grf_hs_t['L']
            initial_r_to, initial_l_to = grf_to_t['R'], grf_to_t['L']
            suggested_r_hs = suggested_l_hs = suggested_r_to = suggested_l_to = []

        # ------------------------------------------------------------------
        # Step 4: Interactive Gait Event GUI -- review, promote ghosts, confirm
        # ------------------------------------------------------------------
        print(f"\n--- Launching Interactive Gait Event GUI for Trial: {name} ---")
        trial.add_corrected_grf(corrected_grf=corrected_grf)

        t_grf = corrected_grf.data['time'].values

        init_speed = selections[0]['speed'] if len(selections) > 0 else 0.0
        init_slope = selections[0]['slope'] if len(selections) > 0 else 0.0

        r_hs_times, l_hs_times, r_to_times, l_to_times, gui_speed, gui_slope = run_interactive_selector(
            trial,
            initial_r_hs=initial_r_hs, initial_l_hs=initial_l_hs,
            initial_r_to=initial_r_to, initial_l_to=initial_l_to,
            suggested_r_hs=suggested_r_hs, suggested_l_hs=suggested_l_hs,
            suggested_r_to=suggested_r_to, suggested_l_to=suggested_l_to,
            speed=init_speed, slope=init_slope, postproc_version=postproc_version
        )

        print(f"\n--- GUI Closed for Trial: {name} ---")
        print(f"  Right foot: {len(r_hs_times)} Heel Strikes, {len(r_to_times)} Toe Offs")
        print(f"  Left foot:  {len(l_hs_times)} Heel Strikes, {len(l_to_times)} Toe Offs")
        print(f"  Trial Speed={gui_speed} mph, Slope={gui_slope}%")

        # Convert confirmed timestamps back to frame indices
        def times_to_indices(times, time_arr):
            indices = []
            for t in times:
                idx = int(np.argmin(np.abs(time_arr - t)))
                indices.append(idx)
            return sorted(list(set(indices)))

        final_hs_moments = {
            'R': times_to_indices(r_hs_times, t_grf),
            'L': times_to_indices(l_hs_times, t_grf)
        }
        final_to_moments = {
            'R': times_to_indices(r_to_times, t_grf),
            'L': times_to_indices(l_to_times, t_grf)
        }

        # ------------------------------------------------------------------
        # Step 5: Baseline correction — applied AFTER events are confirmed
        # ------------------------------------------------------------------
        print(f"\n--- Applying Baseline Correction for Trial: {name} ---")
        post_module.baseline_correct_debug(
            corrected_grf, 'ground_force5_vy', ['ground_force5_vx', 'ground_force5_vz'],
            output_path=local.get_corrected_mot_path(name) if save else None, show=False)
        post_module.baseline_correct_debug(
            corrected_grf, 'ground_force4_vy', ['ground_force4_vx', 'ground_force4_vz'],
            output_path=local.get_corrected_mot_path(name) if save else None, show=False)

        # ------------------------------------------------------------------
        # Step 6: Zero swing phase using finalised events
        # ------------------------------------------------------------------
        print(f"\n--- Zeroing Swing Phase for Trial: {name} ---")
        post_module.zero_swing_phase(corrected_grf, final_to_moments, final_hs_moments, 'right')
        post_module.zero_swing_phase(corrected_grf, final_to_moments, final_hs_moments, 'left')

        if save:
            corrected_grf.save(local.get_corrected_mot_path(name))
            trial.add_corrected_grf(
                corrected_grf=corrected_grf,
                path_to_corrected_grf=os.path.join(local.get_corrected_mot_path(name), corrected_grf.filename))
        else:
            trial.add_corrected_grf(corrected_grf=corrected_grf)

        # Update selections if offset corrector was not used
        if len(selections) == 0:
            selections = [{'tmin': float(t_grf[0]), 'tmax': float(t_grf[-1]), 'speed': gui_speed, 'slope': gui_slope}]

        # ------------------------------------------------------------------
        # Step 7: Segment MOT and TRC at heel strikes
        # ------------------------------------------------------------------
        print(f"\n--- Segmenting MOT and TRC for Trial: {name} ---")
        post_module.segment_at_heel_strikes(
            trial, final_hs_moments,
            save=local.get_segmented_path(name) if save else None)
        print("Segmentation complete. Proceeding with IK -> ID -> JP calculations...")


        # ── Route each cycle's IK / ID / JP outputs into a per-speed subfolder ──
        # Group gait cycles by their speed/slope label using each cycle's time frame.
        from collections import defaultdict
        cycle_groups = {'Right': defaultdict(list), 'Left': defaultdict(list)}

        for side in ['Right', 'Left']:
            for cycle in trial.gait_cycles[side]:
                label = get_cycle_speed_label(cycle, selections)
                cycle_groups[side][label].append(cycle)

        # Log grouping
        for side in ['Right', 'Left']:
            for label, cycles in cycle_groups[side].items():
                print(f"  {side} {label}: {len(cycles)} cycles")

        # Build a temporary mini-trial per speed/slope label and run IK → ID → JP
        for speed_label in set(
            list(cycle_groups['Right'].keys()) + list(cycle_groups['Left'].keys())
        ):
            print(f"\n=== Processing speed group: {speed_label} ===")

            # Create a mini Trial that only holds cycles from this speed window
            mini_trial = Trial(mot=trial.grf, trc=trial.trc, name=f"{name}_{speed_label}")
            mini_trial.gait_cycles = {
                'Right': cycle_groups['Right'].get(speed_label, []),
                'Left':  cycle_groups['Left'].get(speed_label, []),
            }

            total_cycles = (len(mini_trial.gait_cycles['Right']) +
                            len(mini_trial.gait_cycles['Left']))
            if total_cycles == 0:
                print(f"  No cycles found for {speed_label}, skipping.")
                continue

            # Build speed-specific output paths by appending the speed label as a subfolder
            ik_path  = os.path.join(local.get_ik_results_path(name),  speed_label)
            id_path  = os.path.join(local.get_id_results_path(name),  speed_label)
            exl_path = os.path.join(local.get_external_loads_path(name), speed_label)
            jp_path  = os.path.join(local.get_power_filtered_path(name), speed_label)

            compute_ik(mini_trial, local.get_scaled_model_file(), ik_path, save=save)
            compute_id(mini_trial, exl_path, id_path, local.get_scaled_model_file())
            compute_jp(mini_trial, jp_path)

    print("\nAll files were processed.")
