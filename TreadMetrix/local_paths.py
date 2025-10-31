import os
import sys
import json
import tkinter as tk
from tkinter import filedialog

# todo: update this file with tkinter's gui for easier file & directory selections

with open(os.path.join(os.path.dirname(__file__), ".local.json")) as json_loc:
    LOCAL = json.load(json_loc)

_opensim_path = LOCAL['opensim_path']
_base_path = LOCAL['base_path']
_model_file = LOCAL['osim_scaled_model']

# FILL/REPLACE WHERE NEEDED:
frame_rate_mot = 1000  # NUMBER OF FRAMES BY SECOND FOR MOT FILES
frame_rate_trc = 100  # NUMBER OF FRAMES BY SECOND FOR TRC FILES


def configure_opensim():
    os.environ['OPENSIM_HOME'] = _opensim_path
    os.add_dll_directory(_opensim_path)
    sys.path.append(os.path.join(_opensim_path, 'Bindings', 'Python'))
    os.environ['PATH'] += os.pathsep + os.path.join(_opensim_path, 'bin')


def get_rates():
    """
    Return frame rates for MOT files and TRC files.

    Left here until trc.cut.steps is updated with getters to safer frame rates.

    Returns:
        int, frame rate of MOT files
        int, frame rate of TRC files

    """
    return frame_rate_mot, frame_rate_trc


def get_base_path():
    return _base_path


def get_model_file():
    return _model_file


def get_raw_mot_path():
    return LOCAL['raw_mot'] if 'raw_mot' in LOCAL else os.path.join(_base_path, 'raw')


def get_corrected_mot_path():
    return LOCAL['corrected_mot'] if 'corrected_mot' in LOCAL else os.path.join(_base_path, 'corrected_mot')


def get_segmented_mot_path():
    path = LOCAL['segmented_mot'] if 'segmented_mot' in LOCAL else os.path.join(_base_path, 'segmented')
    os.makedirs(path, exist_ok=True)
    return path


def get_raw_trc_path():
    return LOCAL['raw_trc'] if 'raw_trc' in LOCAL else os.path.join(_base_path, 'raw')


def get_segmented_trc_path():
    path = LOCAL['segmented_trc'] if 'segmented_trc' in LOCAL else os.path.join(_base_path, 'segmented')
    os.makedirs(path, exist_ok=True)
    return path


def get_external_loads_path():
    path = LOCAL['external_loads'] if 'external_loads' in LOCAL else os.path.join(_base_path, 'external_loads')
    os.makedirs(path, exist_ok=True)
    return path


def get_ik_results_path():
    path = LOCAL['ik_results'] if 'ik_results' in LOCAL else os.path.join(_base_path, 'ik_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_id_results_path():
    path = LOCAL['id_results'] if 'id_results' in LOCAL else os.path.join(_base_path, 'id_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_power_filtered_path():
    path = LOCAL['power_filtered'] if 'power_filtered' in LOCAL else os.path.join(_base_path, 'power_filtered')
    os.makedirs(path, exist_ok=True)
    return path
