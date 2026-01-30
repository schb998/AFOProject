import os
import re
import time

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

# todo: if "speed" in trial name, ask for input and samples into multiple trials

_static_regex = r"([sS][tT][aA][tT][iI][cC])|([cC][aA][lL])"


def identify_new_trials_from_dict(directory: str, previous_trials: dict[str, Trial] | list[str] | None = None) -> dict[str, Trial]:
    """Identify new trials to process from the given directory.

    Args:
        directory: str,path to the directory in which to search for new trials
        previous_trials: trials already identified

    Returns:
        directory of the trials with their name as the keys

    """
    # list the trials already selected:
    previous_trials_list = previous_trials.keys() if isinstance(previous_trials, dict) \
        else previous_trials if previous_trials is not None \
        else []

    new_trials = {}

    # read the given directory for c3d / mot files:
    for file in os.listdir(directory):

        # exclude the static and calibrations files:
        if re.search(_static_regex, file) is not None:
            pass

        # create a trial from a mot file if a matching trc exists:
        elif file.endswith(MOT.extension):
            trial_name = os.path.basename(file).replace(MOT.extension, '')
            if trial_name not in previous_trials_list:
                mot_path = os.path.join(directory, file)
                trc_path = mot_path.replace(MOT.extension, TRC.extension)
                if os.path.isfile(trc_path):
                    new_trials[trial_name] = Trial(mot=mot_path, trc=trc_path)
                else:
                    pass

        # create a new trial from a c3d file:
        elif file.endswith(".c3d"):
            trial_name = os.path.basename(file).replace('.c3d', '')
            if trial_name not in previous_trials_list:
                new_trials[trial_name] = Trial.from_c3d(os.path.join(directory, file))

    return new_trials


def create_trials_from_saved_selection() -> dict[str, Trial]:
    """Create trials from the paths of the save file

    Returns:
        directory of the trials with their name as the keys

    """
    # read the save file:
    try:
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
    new_trials = {}

    for file in mot_files:
        mot_path = file
        trial_name = os.path.basename(file).replace(MOT.extension, '')
        trc_path = mot_path.replace(MOT.extension, TRC.extension)
        if os.path.isfile(trc_path):
            new_trials[trial_name] = Trial(mot=mot_path, trc=trc_path)
        else:
            print(f"Trial could not be loaded from {trial_name}. Skipping.")
            pass

    for file in c3d_files:
        t = Trial.from_c3d(file)
        new_trials[t.name] = t

    return new_trials


def trials_selection():
    """Select and organize the trials of the pipeline

    Returns:

    """
    try:
        selected_directory = local.get_raw_directory()
        trials_to_process = identify_new_trials_from_dict(selected_directory)
    except MissingPathException:
        selected_directory = None
        trials_to_process = create_trials_from_saved_selection()
    return trials_to_process, selected_directory


def set_up() -> (bool, bool):
    """Sets up the user's preference for the pipeline, including the paths to use and save.

    Return:
        (bool, bool), whether to save the optional files of the pipeline, and whether to show the optional plots

    """
    # update local paths:
    local.main_gui()
    osim.main()
    # ask user's preference
    return local.call_should_save(), local.call_should_show()


def main() -> None:
    """Main pipeline. Runs as long as it's not interrupted, checking for more trials if a directory has been given.

    Returns:
        None

    """
    whether_to_save, whether_to_show = False, False if local.call_quick_setup() else set_up()

    trials, directory = trials_selection()

    trials_processed = {}
    trials_to_process = trials.copy()

    try:
        while True:
            if trials_to_process:
                name = list(trials_to_process.keys())[0]
                trial = trials_to_process.pop(name)
                post_processing(trial, save_plot_path=local.get_corrected_mot_path(name),
                                save_segmented_path=local.get_segmented_path(name) if whether_to_save else None,
                                show=whether_to_show, save_optionals=whether_to_save)
                compute_ik(trial, local.get_scaled_model_file(), local.get_ik_results_path(name),
                           save=whether_to_save)
                compute_id(trial, local.get_external_loads_path(name), local.get_id_results_path(name),
                           local.get_scaled_model_file())
                compute_jp(trial, local.get_power_filtered_path(name))
                trials_processed[name] = trial
                print(f"\nProcessed trial: {name}.")

                # if the user has selected a directory to process, this ensures that the new trials are identified and processed:
                if directory is not None:
                    new_trials = identify_new_trials_from_dict(directory, previous_trials=trials)
                    trials.update(new_trials)
                    trials_to_process.update(new_trials)
                    print(f"\nAdded new trials: {list(new_trials.keys())}." if new_trials else "Processed all trials.")

            # when there is no new trials to process, the program sleeps for 5s and checks if the situation changed:
            else:
                print(f"\nChecking for new trials...")
                new_trials = identify_new_trials_from_dict(directory, previous_trials=trials)
                if new_trials:
                    trials.update(new_trials)
                    trials_to_process.update(new_trials)
                    print(f"Added new trials: {list(new_trials.keys())}.")
                else:
                    print(f"No new trial identified. Waiting...")
                    time.sleep(10)

    # this ensures the user can stop the program:
    except KeyboardInterrupt:
        print(f"\nThe program was interrupted. "
              f"\nTrials processed: {list(trials_processed.keys())}."
              f"\nOther trials: {list(trials_to_process.keys())}.")


if __name__ == "__main__":
    main()

