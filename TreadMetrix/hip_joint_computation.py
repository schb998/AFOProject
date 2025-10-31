import os
import numpy as np
import pandas as pd
from resources.filetypes_gestion.trc import TRC
from copy import deepcopy
import logging


def hjc_harrington(markers):
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
    lasis = np.array(markers['LASIS'])
    rasis = np.array(markers['RASIS'])
    lpsis = np.array(markers['LPSIS'])
    rpsis = np.array(markers['RPSIS'])

    n_frames = rasis.shape[0]

    rhjc = np.zeros((n_frames, 3))
    lhjc = np.zeros((n_frames, 3))

    for t in range(n_frames):
        # Right-handed pelvis reference system definition

        sacrum = (rpsis[t] + lpsis[t]) / 2
        ob = (lasis[t] + rasis[t]) / 2

        provv = (rasis[t] - sacrum) / np.linalg.norm(rasis[t] - sacrum)
        ib = (rasis[t] - lasis[t]) / np.linalg.norm(rasis[t] - lasis[t])
        kb = np.cross(ib, provv)
        kb = kb / np.linalg.norm(kb)
        jb = np.cross(kb, ib)
        jb = jb / np.linalg.norm(jb)

        # Rotation and translation in homogeneous coordinates (4x4)
        pelvis = np.eye(4)
        pelvis[:3, 0] = ib
        pelvis[:3, 1] = jb
        pelvis[:3, 2] = kb
        pelvis[:3, 3] = ob

        # Pelvis width (PelW) and depth (PelD)
        pelw = np.linalg.norm(rasis[t] - lasis[t])
        peld = np.linalg.norm(sacrum - ob)

        # Harrington formulas
        diff_ap = -0.24 * peld - 9.9
        diff_v = -0.30 * pelw - 10.9
        diff_ml = 0.33 * pelw + 7.3

        # Vectors to subtract from OP to get HJC in pelvis coordinate system
        vect_diff_pelvis_sx = np.array([-diff_ml, diff_ap, diff_v, 1])
        vect_diff_pelvis_dx = np.array([diff_ml, diff_ap, diff_v, 1])

        # HJC in pelvis coordinate system
        opb = np.dot(np.linalg.inv(pelvis), np.append(ob, 1))
        rhjc_pelvis = opb + vect_diff_pelvis_dx
        lhjc_pelvis = opb + vect_diff_pelvis_sx

        # Transformation from local to global
        rhjc[t] = np.dot(pelvis[:3, :3], rhjc_pelvis[:3]) + ob
        lhjc[t] = np.dot(pelvis[:3, :3], lhjc_pelvis[:3]) + ob

    return rhjc, lhjc


def add_virtual_markers_to_trc(trc: TRC) -> TRC:
    """
    Copies a TRC object and add virtual hip joints markers (RHJC and LHJC).

    Parameters:
        trc (TRC): The TRC object containing marker data.

    Returns:
        TRC: Updated TRC object with added RHJC and LHJC markers.
    """
    pelvis_marker_names = ['RASIS', 'LASIS', 'RPSIS', 'LPSIS']
    markers = {}
    for m in pelvis_marker_names:
        coordinates = trc.marker_dict[m]
        x = trc.data[coordinates[0]]
        y = trc.data[coordinates[1]]
        z = trc.data[coordinates[2]]
        data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
        markers[m] = data
    rhjc, lhjc = hjc_harrington(markers)

    nb_markers = len(trc.marker_set)
    rhjc_coo = ['X' + str(nb_markers + 1), 'Y' + str(nb_markers + 1), 'Z' + str(nb_markers + 1)]
    lhjc_coo = ['X' + str(nb_markers + 2), 'Y' + str(nb_markers + 2), 'Z' + str(nb_markers + 2)]

    og_markers_nb = deepcopy(trc.metadata['NumMarkers'])

    new = trc.copy()
    new.filename = deepcopy(trc.filename).replace(".trc", "_addedHJ.trc")
    # add markers to set:
    new.marker_set.append('RHJC')
    new.marker_set.append('LHJC')
    new.metadata['origNumMarkers'] = og_markers_nb
    new.metadata['NumMarkers'] = og_markers_nb + 2
    # add marker coordinates to set:
    new.col_names.extend(rhjc_coo)
    new.col_names.extend(lhjc_coo)
    new.marker_dict['RHJC'] = rhjc_coo
    new.marker_dict['LHJC'] = lhjc_coo
    # add data:
    for i in range(3):
        new.data[rhjc_coo[i]] = rhjc[:, i:i+1]
    for i in range(3):
        new.data[lhjc_coo[i]] = lhjc[:, i:i+1]

    return new


def compute_hip_joints(input_path: str, output_path: str) -> TRC:
    """
    Full workflow to read, update, and save a TRC file with new virtual markers.

    Parameters:
        input_path (str): Path to the input TRC file.
        output_path (str): Path to save the updated TRC file.

    Returns:

    """
    try:
        trc = TRC.load(input_path)
        updated_trc = add_virtual_markers_to_trc(trc)
        updated_trc.save(output_path)
        logging.info(f"Updated TRC file saved to: {output_path}")
        return updated_trc
    except Exception as e:
        message = f"Error while processing TRC file: {getattr(e, 'message', repr(e))}"
        logging.warning(message)
        raise Exception(message)


# Example usage
compute_hip_joints(os.path.join("C:\\Users\\lgre690\\Documents\\MyData\\osim_tests", "static_01.trc"),
                   "C:\\Users\\lgre690\\Documents\\MyData\\osim_tests")
