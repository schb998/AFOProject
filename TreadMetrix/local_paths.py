import os
import sys
import json

with open(".local.json") as json_loc:
    LOCAL = json.load(json_loc)

opensim_path = LOCAL['opensim_path']
model_file_name = LOCAL['osim_model_file_name']
base_path = LOCAL['base_path']

# FILL/REPLACE WHERE NEEDED:
frame_rate_mot = 1000  # NUMBER OF FRAMES BY SECOND FOR MOT FILES
frame_rate_trc = 100  # NUMBER OF FRAMES BY SECOND FOR TRC FILES


def configure_opensim():
    os.environ['OPENSIM_HOME'] = opensim_path
    os.add_dll_directory(opensim_path)
    sys.path.append(os.path.join(opensim_path, 'Bindings', 'Python'))
    os.environ['PATH'] += os.pathsep + os.path.join(opensim_path, 'bin')


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
    return base_path


def get_model_file():
    model_folder_path = LOCAL['model_folder_path'] if 'model_folder_path' in LOCAL else base_path
    return os.path.join(model_folder_path, model_file_name)


def get_raw_mot_path():
    return LOCAL['raw_mot_path'] if 'raw_mot_path' in LOCAL else os.path.join(base_path, 'raw')


def get_corrected_mot_path():
    return LOCAL['corrected_mot_path'] if 'corrected_mot_path' in LOCAL else os.path.join(base_path, 'corrected_mot')


def get_segmented_mot_path():
    path = LOCAL['segmented_mot_path'] if 'segmented_mot_path' in LOCAL else os.path.join(base_path, 'segmented\\mot')
    os.makedirs(path, exist_ok=True)
    return path


def get_raw_trc_path():
    return LOCAL['raw_trc_path'] if 'raw_trc_path' in LOCAL else os.path.join(base_path, 'raw')


def get_segmented_trc_path():
    path = LOCAL['segmented_trc_path'] if 'segmented_trc_path' in LOCAL else os.path.join(base_path, 'segmented\\trc')
    os.makedirs(path, exist_ok=True)
    return path


def get_external_loads_path():
    path = LOCAL['external_loads_path'] if 'external_loads_path' in LOCAL else os.path.join(base_path, 'external_loads')
    os.makedirs(path, exist_ok=True)
    return path


def get_ik_results_path():
    path = LOCAL['ik_results_path'] if 'ik_results_path' in LOCAL else os.path.join(base_path, 'ik_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_id_results_path():
    path = LOCAL['id_results_path'] if 'id_results_path' in LOCAL else os.path.join(base_path, 'id_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_power_filtered_path():
    path = LOCAL['power_filtered_path'] if 'power_filtered_path' in LOCAL else os.path.join(base_path, 'power_filtered')
    os.makedirs(path, exist_ok=True)
    return path
