import os

# FILL/REPLACE WHERE NEEDED:
_opensim_path         = r"test"            # PATH TO OPENSIM BINARIES (should end with "\OpenSim 4.5\bin")
_base_path            = r"test"               # PATH TO BASE OUTPUT FOLDER
_osim_model_path      = _base_path  # PATH TO OSIM MODEL FOLDER
_osim_model_file_name = "test"     # NAME OF OSIM MODEL FILE (should end with ".osim")
_raw_mot_path         = os.path.join(_base_path, r"raw_mot")         # PATH TO RAW MOT FILES FOLDER
_raw_trc_path         = os.path.join(_base_path, r"raw_trc")         # PATH TO RAW TRC FILES FOLDER
_frame_rate_mot       = 1000         # NUMBER OF FRAMES BY SECOND FOR MOT FILES
_frame_rate_trc       = 100          # NUMBER OF FRAMES BY SECOND FOR TRC FILES

def get_opensim_path():
    return _opensim_path

def get_base_path():
    return _base_path

def get_osim_path():
    return _osim_model_path

def get_osim_model_file_name():
    return _osim_model_file_name

def get_raw_mot_path():
    return _raw_mot_path

def get_raw_trc_path():
    return _raw_trc_path

def get_frame_rate_mot():
    return _frame_rate_mot

def get_frame_rate_trc():
    return _frame_rate_trc