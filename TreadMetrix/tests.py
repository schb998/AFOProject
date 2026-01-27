import os
import threading
import time

import numpy as np
import pandas as pd

from resources.file_types.mot import MOT, MOTMetadata
from ptb.util.data import Yac3do, MocapDO
from ptb.util.osim.osim_store import OSIMStorage, OsimStorageV1, OSIMStorageV2, HeadersLabels, OSIMForcePlate
from data_postprocessing import process as post_processing
from ik_computing import process as compute_ik
from id_computing import process as compute_id
from joint_power_computing import process as compute_jp

from resources.trial_class import Trial
import GUI

c3d_path = r"C:\Users\lgre690\PycharmProjects\AFOProject\resources\file_types\testing_files\C3D_standard.c3d"
osim_scaled_model = "C:/Users/lgre690/Documents/MyData/ttest/scaled_model_Ella.osim"
output = "C:/Users/lgre690/Documents/MyData/ttest/output"

if __name__ == '__main__':
    """
    c3d = Yac3do(c3d_path)
    c3d_mot = MOT.load_from_c3d(c3d_path)
    mot_mot = MOT.load_from_mot(mot_path)
    """

    """def pipeline(trial_to_process: Trial):
        post_processing(trial_to_process, save_plot_path=output,
                        save_segmented_path=None,
                        show=False, save_optionals=False)
        compute_ik(trial_to_process, osim_scaled_model, output, save=False)
        compute_id(trial_to_process, output, output, osim_scaled_model)
        compute_jp(trial_to_process, output)

    trial = Trial(mot=mot_path, trc=trc_path, name="test_trial")

    pipeline_thread = threading.Thread(target=lambda:{pipeline(trial)}, daemon=True)
    gui_thread = threading.Thread(target=lambda: {}, daemon=True)

    pipeline_thread.run()
    gui_thread.run()"""

    # trial1 = Trial(mot=r"C:\Users\lgre690\Documents\MyData\ttest\raw\walking_Incline.mot", trc=r"C:\Users\lgre690\Documents\MyData\ttest\raw\walking_Incline.trc")
    # trial2 = Trial(mot=r"C:\Users\lgre690\Documents\MyData\ttest\raw\walking_incline_NoAFO.mot", trc=r"C:\Users\lgre690\Documents\MyData\ttest\raw\walking_incline_NoAFO.trc")
    # GUI.main(output, osim_scaled_model, {trial1.name: trial1, trial2.name: trial2})



print("All done.")
