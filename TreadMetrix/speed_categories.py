from copy import deepcopy

import pandas as pd

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
import data_postprocessing as pp
import numpy as np
import resources.paths.paths_access as local


# tests to visualize the difference of speed in a unique file

def segment():
    # local.main_gui()

    mot_object = MOT.load_from_mot(local.get_raw_mot_path()[0])
    trc_object = TRC.load_from_trc(local.get_raw_trc_path()[0])

    frame_rate = 1 / np.mean(np.diff(mot_object.data['time']))

    pp.filter_grf(mot_object, frame_rate)
    pp.baseline_correct_debug(mot_object, 'ground_force2_vy', ['ground_force2_vx', 'ground_force2_vz'])
    pp.baseline_correct_debug(mot_object, 'ground_force1_vy', ['ground_force1_vx', 'ground_force1_vz'])
    mot_to_moments = pp.detect_toe_offs(mot_object, frame_rate)
    mot_hs_moments = pp.detect_heel_strikes(mot_object, frame_rate)
    pp.zero_swing_phase(mot_object, mot_to_moments, mot_hs_moments, 'right')
    pp.zero_swing_phase(mot_object, mot_to_moments, mot_hs_moments, 'left')

    r_hs = mot_hs_moments["R"]
    l_hs = mot_hs_moments["L"]
    r_hs_trc = list(np.array(r_hs) / 10)
    l_hs_trc = list(np.array(l_hs) / 10)

    hs_data = np.array(mot_object.data.iloc[r_hs][1:-1])
    """
    l_mot_hs_data = mot_object.data.iloc[l_hs][1:-1]
    r_trc_hs_data = trc_object.data.iloc[r_hs_trc][1:-1]
    l_trc_hs_data = trc_object.data.iloc[l_hs_trc][1:-1]
    """

    mins = []
    for axis in range(hs_data.shape[1]):
        mins.append(np.min(hs_data[:, axis]))

    m_hs = np.zeros(hs_data.shape)
    for axis in range(hs_data.shape[1]):
        for col in range(hs_data.shape[0]):
            m_hs[col, axis] = hs_data[col, axis] - mins[axis]

    m_hs = pd.DataFrame(m_hs)

    print("okay")


    """
    res = pp.segment_at_heel_strikes(mot_object, heel_strike_moments, mot_frame_rate=frame_rate,
                                     trc=TRC.load_from_trc(trc_object), save=local.get_segmented_path())
    return res['mot'], res['trc']
    """


if __name__ == "__main__":
    segment()

    print("okay")
