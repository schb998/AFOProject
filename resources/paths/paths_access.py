import os
from resources.custom_exceptions import *
import resources.paths.paths_back as model
from resources.paths.paths_gui import main as main_gui
from resources.tkinter_toolbox import ask_question


def call_to_gui() -> None:
    main_gui()


def call_quick_setup() -> bool:
    return ask_question("Run the pipeline using quick setup?", "Quick setup")

def call_should_save() -> bool:
    return ask_question("Save the optional files?", "Saving preference")

def call_should_show() -> bool:
    return ask_question("Show the plots as the code run?", "Show preference")

def get_output_path(trial_name: str = None) -> str:
    """Get the pipeline's saved output path.

    Args:
        trial_name (str, optional): Trial name. Defaults to None.

    Returns:
        str, stored output path

    Raises:
        MissingPathException if no such path has been given

    """
    content = model.get_local("output_path")
    if content is None:
        raise MissingPathException("Output directory")
    if trial_name is not None:
        content = os.path.join(content, trial_name)
        os.makedirs(content, exist_ok=True)
    return content


def get_osim_path() -> str | None:
    """Get the pipeline's saved OpenSim binaries path.

    Returns:
        str, stored path to OpenSim binaries directory

    Raises:
        MissingPathException if no such path has been given
    """
    content = model.get_local("osim_path")
    if content is None:
        raise MissingPathException("Opensim binaries")
    return content


def get_scaled_model_file() -> str:
    content = model.get_local("osim_scaled_model")
    if content is not None:
        return content
    raise MissingPathException("scaled OpenSim model")


def get_base_model_file() -> str:
    content = model.get_local("osim_base_model")
    if content is not None:
        return content
    raise MissingPathException("base OpenSim model")


def get_raw_directory() -> str:
    content = model.get_local("raw_directory")
    if content is not None:
        return content
    raise MissingPathException("raw directory")


def get_raw_mot_path() -> list[str]:
    path = model.get_local("raw_mot")
    if path is not None:
        return path
    raise MissingPathException("list of raw mots to process")


def get_raw_trc_path() -> list[str]:
    path = model.get_local("raw_trc")
    if path is not None:
        return path
    raise MissingPathException("list of raw trcs to process")


def get_corrected_mot_path(trial_name: str = None) -> str:
    path = os.path.join(get_output_path(trial_name), 'corrected_mot')
    os.makedirs(path, exist_ok=True)
    return path


def get_segmented_path(trial_name: str = None) -> str:
    path = os.path.join(get_output_path(trial_name), 'segmented')
    os.makedirs(path, exist_ok=True)
    return path


def get_external_loads_path(trial_name: str = None) -> str:
    path = os.path.join(get_output_path(trial_name), 'external_loads')
    os.makedirs(path, exist_ok=True)
    return path


def get_ik_results_path(trial_name: str = None) -> str:
    path = os.path.join(get_output_path(trial_name), 'ik_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_id_results_path(trial_name: str = None) -> str:
    path = os.path.join(get_output_path(trial_name), 'id_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_power_filtered_path(trial_name: str = None) -> str:
    path = os.path.join(get_output_path(trial_name), 'power_filtered')
    os.makedirs(path, exist_ok=True)
    return path

