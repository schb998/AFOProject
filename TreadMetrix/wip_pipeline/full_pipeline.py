import os
import re
from tkinter import messagebox
import resources.paths.paths_access as local
from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
from resources.trial_class import Trial
import osim_gestion as osim
from data_postprocessing import process as post_processing
from ik_computing import process as compute_ik
from id_computing import process as compute_id


if __name__ == "__main__":

    # quick setup for debug
    if local.call_quick_setup():
        save = False
        show = False

    else:
        # update local paths:
        local.main_gui()
        osim.configure_opensim()
        # ask user's preference
        save = messagebox.askokcancel("Save optional files", "Save optional files")
        show = messagebox.askokcancel("Show plots when running", "Show plots on screen during processing")

    # loads files into Trial objects:
    trials = {}
    for file in local.get_raw_mot_path():
        try:
            mot = MOT.load_from_mot(file)
            trial_name = mot.filename.replace('.mot', '')
            try:
                trc = [t for t in local.get_raw_trc_path() if re.search(trial_name + r"\.trc$", t) is not None][0]
            except IndexError:
                raise OSError
            trials[trial_name] = Trial(mot, trc=TRC.load_from_trc(trc))
        except OSError:
            print(f"Trial could not be loaded from {file} couldn't be loaded. Skipping.")
            break

    # process the trials:
    for name in trials:
        trial = trials[name]


        post_processing(trial, save_corrected_path=os.path.join(local.get_corrected_mot_path(), name) if save else None,
                        save_segmented_path=os.path.join(local.get_segmented_path(), name) if save else None, show=show)

        trial = trial.sample(15.0, 30.0)

        compute_ik(trial, local.get_scaled_model_file(), os.path.join(local.get_ik_results_path(), name), save=save)
        compute_id(trial, os.path.join(local.get_external_loads_path(), name),
                   os.path.join(local.get_id_results_path(), name), local.get_scaled_model_file())

    print("\nAll files were processed.")
