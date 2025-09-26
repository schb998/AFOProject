import os
import local as loc

opensim_path = loc.get_opensim_path()
base_path = loc.get_base_path()
model_folder_path = loc.get_osim_path()
model_file_name = loc.get_osim_model_file_name()
raw_mot_path = loc.get_raw_mot_path()
raw_trc_path = loc.get_raw_trc_path()
frame_rate_mot = loc.get_frame_rate_mot()
frame_rate_trc = loc.get_frame_rate_trc()

def configure_opensim():
    os.environ['OPENSIM_HOME'] = opensim_path
    os.add_dll_directory(opensim_path)

def get_rates():
    """
    Return frame rates fo MOT files and TRC files.

    Returns:
        int, frame rate of MOT files
        int, frame rate of TRC files

    """
    return frame_rate_mot, frame_rate_trc

def get_base_path():
    return base_path

def get_model_file():
    return os.path.join(model_folder_path, model_file_name)

def get_raw_mot_path():
    return raw_mot_path

def get_corrected_mot_path():
    return os.path.join(base_path, "corrected_mot")

def get_segmented_mot_path():
    return os.path.join(base_path, "segmented_mot")

def get_raw_trc_path():
    return raw_trc_path

def get_segmented_trc_path():
    return os.path.join(base_path, "segmented_trc")

def get_external_loads_path():
    return os.path.join(base_path, "external_loads")

def get_ik_results_path():
    return os.path.join(base_path, "IK_results")

def get_id_results_path():
    return os.path.join(base_path, "ID_results")

def get_power_filtered_path():
    return os.path.join(base_path, "Power_Filtered")