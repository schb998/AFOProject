import numpy as np
import os
import re
from tkinter import messagebox

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
import resources.paths.paths_access as local
from resources.paths.paths_gui import main as gui_main
import data_postprocessing as pp
import osim_gestion as osim
import ik_data as ik


if __name__ == "__main__":
    # update local paths and read them:
    gui_main()
    osim.configure_opensim()

    raw_mot_files = local.get_raw_mot_path()
    raw_trc_files = local.get_raw_trc_path()
    mot_corrected_output = local.get_corrected_mot_path()

    save = messagebox.askokcancel("Save optional files", "Save optional files")
    show = messagebox.askokcancel("Show plots when running", "Show plots on screen during processing")

    # loads files:
    results = {}
    for file in raw_mot_files:
        try:
            m = MOT.load(file)
        except OSError:
            print(f"File {file} couldn't be loaded. Skipping.")
            break
        results[m.filename.replace('.mot', '')] = {'mot': m}
    raw_mot_files = results.keys()

    # process files:
    for name in raw_mot_files:
        m = results[name]['mot']
        time = m.data['time']
        frame_rate = 1 / np.mean(np.diff(time))
        print(f"\nProcessing: {m.filename} with sampling frequency: {frame_rate:.2f} Hz.")
        save_corrected_path = os.path.join(mot_corrected_output, name)

        # apply filters and baseline correction:
        pp.filter_grf(m, frame_rate)
        pp.baseline_correct_debug(m, 'ground_force2_vy', ['ground_force2_vx', 'ground_force2_vz'],
                                  save_corrected_path, show=show)
        pp.baseline_correct_debug(m, 'ground_force1_vy', ['ground_force1_vx', 'ground_force1_vz'],
                                  save_corrected_path, show=show)
        toe_off_moments = pp.detect_toe_offs(m, frame_rate)
        heel_strike_moments = pp.detect_heel_strikes(m, frame_rate)
        right_mot = m.copy()
        pp.zero_swing_phase(m, toe_off_moments, heel_strike_moments, 'right')
        pp.zero_swing_phase(m, toe_off_moments, heel_strike_moments, 'left')
        m.rename(name=m.filename.replace('.mot', '') + "_corrected",
                 filename=m.filename.replace('.mot', '_corrected.mot'), )

        # pp.plot_grf_details(m, heel_strike_moments, toe_off_moments, str(save_corrected_path), show=show)
        m.save(save_corrected_path)

        # segment according to heel strikes:
        trc = None
        trc_name_regex = name + "\\.trc$"
        trc_found = False

        try:
            matching_trc = [t for t in raw_trc_files if re.search(trc_name_regex, t) is not None][0]
            trc_found = True
        except IndexError:
            print(f"No selected TRC file matching MOT file {name}. Skipping.")
            matching_trc = None

        if trc_found:
            try:
                TRC.adapt_to_opensim_use(matching_trc)
                trc = TRC.load_from_trc(matching_trc)
                results[name]['segmented'] = pp.segment_at_heel_strikes(m, heel_strike_moments, mot_frame_rate=frame_rate,
                                                                        trc=trc, save=save)
            except OSError:
                print("Could not TRC load")
                trc_found = False

        if not trc_found:
            results[name]['segmented'] = pp.segment_at_heel_strikes(m, heel_strike_moments, save=save)

        else:
            print("IK postprocessing part fo the pipeline is still a WIP.")
            # ik.process(results[name]['segmented']['trc'], name)

    print("\nAll files were processed.")
