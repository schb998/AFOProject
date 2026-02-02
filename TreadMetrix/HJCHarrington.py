import os
import numpy as np
import pandas as pd
from yatpkg.util.data import TRC as YatTRC
from copy import deepcopy

def read_trc_files(data_path_trc):
    """
    Reads all TRC files in the given directory.

    Parameters:
        data_path_trc (str): Path to the directory containing TRC files.

    Returns:
        list: List of YatTRC objects for the TRC files.
    """
    if not os.path.isdir(data_path_trc):
        print(f"The provided path '{data_path_trc}' is not a directory.")
        return []

    trc_data_list = []
    for filename in os.listdir(data_path_trc):
        if filename.endswith('.trc'):
            file_path = os.path.join(data_path_trc, filename)
            print(f"Reading TRC file: {file_path}")
            try:
                trc_data = YatTRC.read(file_path)
                trc_data_list.append(trc_data)
                print(f"Successfully read {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return trc_data_list

def HJCHarrington(markers):
    """
    Calculate Hip Joint Centers (HJC) using Harrington et al. (2006) method.

    Parameters:
    markers: dict
        Dictionary containing the trajectories of the required markers from a static trial.
        Keys: 'LASIS', 'RASIS', 'LPSIS', 'RPSIS' (numpy arrays of shape [n_frames, 3]).

    Returns:
    RHJC: numpy array
        Right hip joint center positions [n_frames, 3].
    LHJC: numpy array
        Left hip joint center positions [n_frames, 3].
    """
    LASIS = markers['LASIS']
    RASIS = markers['RASIS']
    LPSIS = markers['LPSIS']
    RPSIS = markers['RPSIS']

    n_frames = RASIS.shape[0]

    RHJC = np.zeros((n_frames, 3))
    LHJC = np.zeros((n_frames, 3))

    for t in range(n_frames):
        # Right-handed pelvis reference system definition
        SACRUM = (RPSIS[t] + LPSIS[t]) / 2
        OP = (LASIS[t] + RASIS[t]) / 2

        PROVV = (RASIS[t] - SACRUM) / np.linalg.norm(RASIS[t] - SACRUM)
        IB = (RASIS[t] - LASIS[t]) / np.linalg.norm(RASIS[t] - LASIS[t])
        KB = np.cross(IB, PROVV)
        KB = KB / np.linalg.norm(KB)
        JB = np.cross(KB, IB)
        JB = JB / np.linalg.norm(JB)

        OB = OP

        # Rotation and translation in homogeneous coordinates (4x4)
        pelvis = np.eye(4)
        pelvis[:3, 0] = IB
        pelvis[:3, 1] = JB
        pelvis[:3, 2] = KB
        pelvis[:3, 3] = OB

        # Pelvis width (PW) and depth (PD)
        PW = np.linalg.norm(RASIS[t] - LASIS[t])
        PD = np.linalg.norm(SACRUM - OP)

        # Harrington formulas
        diff_ap = -0.24 * PD - 9.9
        diff_v = -0.30 * PW - 10.9
        diff_ml = 0.33 * PW + 7.3

        # Vectors to subtract from OP to get HJC in pelvis coordinate system
        vect_diff_pelvis_sx = np.array([-diff_ml, diff_ap, diff_v, 1])
        vect_diff_pelvis_dx = np.array([diff_ml, diff_ap, diff_v, 1])

        # HJC in pelvis coordinate system
        OPB = np.dot(np.linalg.inv(pelvis), np.append(OB, 1))
        rhjc_pelvis = OPB + vect_diff_pelvis_dx
        lhjc_pelvis = OPB + vect_diff_pelvis_sx

        # Transformation from local to global
        RHJC[t] = np.dot(pelvis[:3, :3], rhjc_pelvis[:3]) + OB
        LHJC[t] = np.dot(pelvis[:3, :3], lhjc_pelvis[:3]) + OB

    return RHJC, LHJC

def add_virtual_markers_to_trc(trc):
    """
    Adds virtual markers (RHJC and LHJC) to a TRC object.

    Parameters:
        trc (YatTRC): The TRC object containing marker data.

    Returns:
        YatTRC: Updated TRC object with added RHJC and LHJC markers.
    """
    pelvis_marker_names = ['RASIS', 'LASIS', 'RPSIS', 'LPSIS']
    markers = {
        marker: trc.marker_set[marker].values for marker in pelvis_marker_names
    }

    RHJC, LHJC = HJCHarrington(markers)

    trc.marker_set['RHJC'] = pd.DataFrame(RHJC, columns=['X28', 'Y28', 'Z28'])
    trc.marker_set['LHJC'] = pd.DataFrame(LHJC, columns=['X29', 'Y29', 'Z29'])
    trc.marker_names.extend(['RHJC', 'LHJC'])
    trc.update_from_markerset()
    trc.headers['Units'] = 'mm'
    return trc

def process_trc(input_path, output_path):
    """
    Full workflow to read, update, and save a TRC file with new virtual markers.

    Parameters:
        input_path (str): Path to the input TRC file.
        output_path (str): Path to save the updated TRC file.
    """
    try:
        trc = YatTRC.read(input_path)
        trc = add_virtual_markers_to_trc(trc)
        trc.write(output_path)
        print(f"Updated TRC file saved to: {output_path}")
    except Exception as e:
        print(f"Error while processing TRC file: {e}")

# Example usage
process_trc(
    "C:/Users/schb998/MyData/Pilot_Ella/Static.trc/static_01.trc",
    "C:/Users/schb998/MyData/Pilot_Ella/updated_static.trc"
)

