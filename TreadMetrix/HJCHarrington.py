import os
import numpy as np
import pandas as pd
from yatpkg.util.data import TRC as YatTRC, StorageIO
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
                if file_path.lower().endswith('.trc'):
                    trc_data = StorageIO.trc_reader(file_path)
                elif file_path.lower().endswith('.c3d'):
                    trc_data = YatTRC.create_from_c3d(file_path)
                else:
                    print(f"Unsupported file format: {filename}")
                    continue
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
    # Check for alternative marker names
    def get_marker_name(base_name, trc_names):
        if base_name in trc_names:
            return base_name
        # If the base name has 'S' at the end, try without it, or vice versa
        alt_name = base_name[:-1] if base_name.endswith('S') else base_name + 'S'
        if alt_name in trc_names:
            return alt_name
        raise ValueError(f"Marker {base_name} or {alt_name} not found in TRC file")

    pelvis_marker_names = ['RASIS', 'LASIS', 'RPSIS', 'LPSIS']
    actual_names = [get_marker_name(name, trc.marker_names) for name in pelvis_marker_names]
    
    markers = {
        expected: trc.marker_set[actual].values for expected, actual in zip(pelvis_marker_names, actual_names)
    }

    RHJC, LHJC = HJCHarrington(markers)

    inx = (trc.data.shape[1] - 2) // 3 + 1
    trc.marker_set['RHJC'] = pd.DataFrame(RHJC, columns=[f'X{inx}', f'Y{inx}', f'Z{inx}'])
    trc.column_labels.extend([f'RHJC_X{inx}', f'RHJC_Y{inx}', f'RHJC_Z{inx}'])
    inx += 1
    trc.marker_set['LHJC'] = pd.DataFrame(LHJC, columns=[f'X{inx}', f'Y{inx}', f'Z{inx}'])
    trc.column_labels.extend([f'LHJC_X{inx}', f'LHJC_Y{inx}', f'LHJC_Z{inx}'])
    
    trc.marker_names.extend(['RHJC', 'LHJC'])
    
    new_data = np.hstack((RHJC, LHJC))
    trc.data = np.hstack((trc.data, new_data))
    
    if 'NumMarkers' in trc.headers:
        trc.headers['NumMarkers'] = str(int(trc.headers['NumMarkers']) + 2)
    else:
        # Some versions might use the enum value directly, or it might be missing
        for key in trc.headers:
            if str(key) == 'NumMarkers':
                trc.headers[key] = str(int(trc.headers[key]) + 2)
    
    trc.headers['Units'] = 'mm'
    trc.st = 0  # Force TRC writer to regenerate the headers with the new markers
    return trc

def rotate_to_opensim(trc):
    """
    Rotates the TRC data from the current mocap coordinate system (X=Left, Y=Back, Z=Up)
    to the OpenSim coordinate system (X=Forward, Y=Up, Z=Right).
    Transformation: X_os = -Y, Y_os = Z, Z_os = -X
    """
    for i in range(2, trc.data.shape[1], 3):
        X_old = trc.data[:, i].copy()
        Y_old = trc.data[:, i+1].copy()
        Z_old = trc.data[:, i+2].copy()
        
        trc.data[:, i] = -Y_old
        trc.data[:, i+1] = Z_old
        trc.data[:, i+2] = -X_old
        
    for m in trc.marker_names:
        marker_df = trc.marker_set[m]
        X_old = marker_df.iloc[:, 0].copy()
        Y_old = marker_df.iloc[:, 1].copy()
        Z_old = marker_df.iloc[:, 2].copy()
        
        marker_df.iloc[:, 0] = -Y_old
        marker_df.iloc[:, 1] = Z_old
        marker_df.iloc[:, 2] = -X_old
        
    return trc

def process_trc(input_path, output_path):
    """
    Full workflow to read, update, and save a TRC file with new virtual markers.

    Parameters:
        input_path (str): Path to the input TRC file.
        output_path (str): Path to save the updated TRC file.
    """
    if input_path.lower().endswith('.trc'):
        trc = StorageIO.trc_reader(input_path)
    elif input_path.lower().endswith('.c3d'):
        trc = YatTRC.create_from_c3d(input_path)
    else:
        print(f"Unsupported file format: {input_path}")
        return
        
    trc = add_virtual_markers_to_trc(trc)
    trc.write(output_path)
    print(f"Updated TRC file saved to: {output_path}")

# Example usage
input_file = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\Gait01\P03 Cal 02.trc"
output_file = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\Gait01\P03 Cal 02_updated.trc"
process_trc(input_file, output_file)

