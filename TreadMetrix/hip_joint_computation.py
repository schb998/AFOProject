import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from resources.file_types.trc import TRC
from copy import deepcopy
import logging
import re


def hjc_harrington(marker_data: dict[str, npt.ArrayLike]) -> (np.typing.ArrayLike, np.typing.ArrayLike):
    """
    Calculate Hip Joint Centers (HJC) using Harrington et al. (2006) method.

    Parameters: marker_data: dictionary of the pelvis markers' trajectories, by marker (marker_name = numpy arrays [
    n_frames, 3]).

    Returns:
    RHJC: numpy array
        Right hip joint center positions [n_frames, 3].
    LHJC: numpy array
        Left hip joint center positions [n_frames, 3].
    """
    marker_names = list(marker_data.keys())
    lasis = np.array(marker_data[marker_names[0]]).astype(float)
    lpsis = np.array(marker_data[marker_names[1]]).astype(float)
    rasis = np.array(marker_data[marker_names[2]]).astype(float)
    rpsis = np.array(marker_data[marker_names[3]]).astype(float)

    n_frames = lasis.shape[0]

    rhjc = np.zeros((n_frames, 3))
    lhjc = np.zeros((n_frames, 3))

    for t in range(n_frames):
        # Right-handed pelvis reference system definition

        sacrum = (rpsis[t] + lpsis[t]) / 2
        ob = (lasis[t] + rasis[t]) / 2

        provv = (rasis[t] - sacrum) / np.linalg.norm(rasis[t] - sacrum)
        ib = (rasis[t] - lasis[t]) / np.linalg.norm(rasis[t] - lasis[t])
        kb = np.cross(ib, provv) # todo: issue here for data from C3D!!
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
    old_name = trc.filename

    rasi = [m for m in trc.marker_set if re.search("^RASI", m) is not None][0]
    lasi = [m for m in trc.marker_set if re.search("^LASI", m) is not None][0]
    rpsi = [m for m in trc.marker_set if re.search("^RPSI", m) is not None][0]
    lpsi = [m for m in trc.marker_set if re.search("^LPSI", m) is not None][0]
    pelvis_marker_names = [lasi, lpsi, rasi, rpsi]

    markers = {}
    pelvis_marker_names.sort()
    for m in pelvis_marker_names:
        coordinates = trc.marker_dict[m]
        x = trc.data[coordinates[0]]
        y = trc.data[coordinates[1]]
        z = trc.data[coordinates[2]]
        data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
        markers[m] = data
    rhjc, lhjc = hjc_harrington(markers)

    new = trc.copy()
    new.add_marker("RHJC", rhjc)
    new.add_marker("LHJC", lhjc)
    new.rename(filename=old_name)

    return new


def compute_hip_joints(input_path: str = None, input_object: TRC = None, output_path: str = None) -> TRC:
    """
    Full workflow to read, update, and save a TRC file with new virtual markers.

    Parameters:
        input_object:
        input_path (str): Path to the input TRC file.
        output_path (str): Path to save the updated TRC file. Optional. Object is not saved if not indicated.

    Returns:
        TRC object with added hip joint markers.

    """
    try:
        trc = TRC.load_from_trc(input_path) if input_object is None else input_object
        updated_trc = add_virtual_markers_to_trc(trc)
        TRC.adapt_to_opensim_use(trc=updated_trc, output_path=output_path)
        if output_path is not None:
            updated_trc.update_data(filepath=output_path)
        logging.info(f"Updated TRC file saved to: {updated_trc.filepath}")
        return updated_trc
    except Exception as e:
        message = f"Error while processing TRC file: {getattr(e, 'message', repr(e))}"
        logging.warning(message)
        raise Exception(message)