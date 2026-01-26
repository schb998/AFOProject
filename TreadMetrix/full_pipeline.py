import os
import re
import resources.paths.paths_access as local
from resources.custom_exceptions import MissingPathException
from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
from resources.trial_class import Trial
import osim_gestion as osim
from data_postprocessing import process as post_processing
from ik_computing import process as compute_ik
from id_computing import process as compute_id
from joint_power_computing import process as compute_jp

_static_regex = r"([sS][tT][aA][tT][iI][cC])|([cC][aA][lL])"

if __name__ == "__main__":

    # quick setup for debug
    if local.call_quick_setup():
        save = False
        show = False

    else:
        # update local paths:
        local.main_gui()
        osim.main()
        # ask user's preference
        save = local.call_should_save()
        show = local.call_should_show()

    try:
        directory = local.get_raw_directory()
        mot_files = [os.path.join(directory, file) for file in os.listdir(directory)
                     if file.endswith(MOT.extension) and re.search(_static_regex, file) is None]
        trc_files = [os.path.join(directory, file) for file in os.listdir(directory)
                    if file.endswith(TRC.extension) and re.search(_static_regex, file) is None]
        c3d_files = [os.path.join(directory, file) for file in os.listdir(directory)
                     if file.endswith(".c3d") and re.search(_static_regex, file) is None]
    except MissingPathException:
        try :
            mot_files = local.get_raw_mot_path()
        except MissingPathException:
            mot_files = []
        try:
            trc_files = local.get_raw_trc_path()
        except MissingPathException:
            trc_files = []
        try:
            c3d_files = local.get_raw_c3d_path()
        except MissingPathException:
            c3d_files = []

    mot_files.sort()
    trc_files.sort()

    # loads files into Trial objects:
    trials = {}

    for file in mot_files:
        trial_name = os.path.basename(file).replace('.mot', '')
        try:
            trial = Trial(mot=file)
            try:
                trial.add_trc([t for t in trc_files if re.search(trial_name + r"\.trc$", t) is not None][0])
            except IndexError:
                raise OSError
            trials[trial_name] = trial
        except OSError:
            print(f"Trial could not be loaded from {trial_name}. Skipping.")
            break

    for file in c3d_files:
        trial = Trial.from_c3d(file)
        trials[trial.name] = trial

    # process the trials:
    for name in trials:
        trial = trials[name]

        post_processing(trial, save_plot_path=local.get_corrected_mot_path(name),
                        save_segmented_path=local.get_segmented_path(name) if save else None,
                        show=show, save_optionals=save)

        compute_ik(trial, local.get_scaled_model_file(), local.get_ik_results_path(name), save=save)
        compute_id(trial, local.get_external_loads_path(name), local.get_id_results_path(name), local.get_scaled_model_file())
        compute_jp(trial, local.get_power_filtered_path(name))

    print("\nAll files were processed.")
