import numpy as np
import os

from resources.filetypes_gestion.mot import MOT
from resources.filetypes_gestion.trc import TRC
import TreadMetrix.local_paths as local
import data_postprocessing as pp

if __name__ == "__main__":
    mot_raw_data_path = local.get_raw_mot_path()
    trc_raw_data_path = local.get_raw_trc_path()
    mot_corrected_output = local.get_corrected_mot_path()

    # loads mot files:
    mot_file_list = sorted(f for f in os.listdir(mot_raw_data_path) if f.endswith('.mot'))
    mot_file_list = [file for file in mot_file_list if "static" not in file.lower()]
    results = {}
    for file in mot_file_list:
        try:
            m = MOT.load(mot_raw_data_path, file)
        except OSError:
            print(f"File {file} couldn't be loaded. Skipping.")
            break
        results[m.filename.replace('.mot', '')] = {'mot': m}
    mot_file_list = results.keys()

    # process files:
    for name in mot_file_list:
        m = results[name]['mot']
        time = m.data['time']
        frame_rate = 1 / np.mean(np.diff(time))
        print(f"\nProcessing: {m.filename} with sampling frequency: {frame_rate:.2f} Hz.")
        save_corrected_path = os.path.join(mot_corrected_output, name)

        # apply filters and baseline correction:
        pp.filter_grf(m, frame_rate)
        pp.baseline_correct_debug(m, 'ground_force2_vy', ['ground_force2_vx', 'ground_force2_vz'],
                                  save_corrected_path)
        pp.baseline_correct_debug(m, 'ground_force1_vy', ['ground_force1_vx', 'ground_force1_vz'],
                                  save_corrected_path)
        toe_off_moments = pp.detect_toe_offs(m, frame_rate)
        heel_strike_moments = pp.detect_heel_strikes(m, frame_rate)
        right_mot = m.copy()
        pp.zero_swing_phase(m, toe_off_moments, heel_strike_moments, 'right')
        pp.zero_swing_phase(m, toe_off_moments, heel_strike_moments, 'left')
        m.rename(name=m.filename.replace('.mot', '') + "_corrected",
                 filename=m.filename.replace('.mot', '_corrected.mot'), )

        # pp.plot_grf_details(m, heel_strike_moments, toe_off_moments, str(save_corrected_path))
        m.save(save_corrected_path)

        # segment according to heel strikes:
        trc = None
        try:
            trc = TRC.load(trc_raw_data_path, m.filename.replace('_corrected.mot', '.trc'))
        except OSError:
            print(f"No TRC file in {trc_raw_data_path} matching MOT file {name}. Skipping.")

        if trc is None:
            results[name]['segmented'] = pp.segment_at_heel_strikes(m, heel_strike_moments, save=False)
        else:
            results[name]['segmented'] = pp.segment_at_heel_strikes(m, heel_strike_moments, mot_frame_rate=frame_rate,
                                                                    trc=trc, save=False)
    print("\nAll files were processed.")
