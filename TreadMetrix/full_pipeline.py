import numpy as np
import os

from resources.filetypes_gestion.mot import MOT
from resources.filetypes_gestion.trc import TRC
import local_paths as local

import data_postprocessing as pp

if __name__ == "__main__":
    # loads mot files:
    mot_raw_data_path = local.get_raw_mot_path()
    mot_output        = local.get_corrected_mot_path()
    os.makedirs(mot_output, exist_ok=True)
    mot_file_list = sorted(f for f in os.listdir(mot_raw_data_path) if f.endswith('.mot'))
    mot_files     = [file for file in mot_file_list if not "static" in file.lower()]
    mots = []
    for file in mot_files:
        mots.append(MOT.load(mot_raw_data_path, file))

    # loads trc files
    trc_raw_data_path = local.get_raw_trc_path()
    trc_file_list = sorted(f for f in os.listdir(trc_raw_data_path) if f.endswith('.trc'))
    trc_files = [file for file in trc_file_list if not "static" in file.lower()]
    trcs = []
    for file in trc_files:
        trcs.append(TRC.load(trc_raw_data_path, file))

    segmented_mots = {}

    # process mot files:
    for m in mots:
        time       = m.data['time']
        frame_rate = 1 / np.mean(np.diff(time))
        print(f"\nProcessing: {m.filename} with sampling frequency: {frame_rate:.2f} Hz.")

        # apply filters and baseline correction:
        pp.filter_grf(m, frame_rate)
        pp.baseline_correct_debug(m, 'ground_force2_vy', ['ground_force2_vx', 'ground_force2_vz'], mot_output)
        pp.baseline_correct_debug(m, 'ground_force1_vy', ['ground_force1_vx', 'ground_force1_vz'], mot_output)
        toe_off_moments     = pp.detect_toe_offs(m, frame_rate)
        heel_strike_moments = pp.detect_heel_strikes(m, frame_rate)
        pp.zero_swing_phase(m, toe_off_moments, heel_strike_moments, 'right')
        pp.zero_swing_phase(m, toe_off_moments, heel_strike_moments, 'left')
        m.rename(name     = m.filename.replace('.mot', '') + "_corrected",
                 filename = m.filename.replace('.mot', '_corrected.mot'), )
        pp.plot_grf_details(m, heel_strike_moments, toe_off_moments, mot_output)
        m.save(mot_output)

        # segment MOT according to heel strikes
        # segmented_mots[m.name] = m.segment(heel_strike_moments)[1:-1]

    print("\n All files were processed.")