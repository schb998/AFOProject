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

def call_should_use_offset_corrector() -> bool:
    val = model.get_local("use_offset_corrector")
    if val is not None:
        return bool(val)
    return ask_question("Apply interactive treadmill offset corrector?", "Offset Corrector preference")

def call_postprocessing_version() -> str:
    val = model.get_local("postprocessing_version")
    if val in ["v1", "v2"]:
        return str(val).lower()
    use_v2 = ask_question("Use Data Post-Processing V2 (Stance Boundary Detection)? Select 'No' for V1 (Peak Detection).", "Post-Processing Version")
    return "v2" if use_v2 else "v1"

def call_should_use_interactive_selector() -> bool:
    val = model.get_local("use_interactive_gait_selector")
    if val is not None:
        return bool(val)
    return ask_question("Use interactive Gait Event & TRC/MOT Segmenter GUI?", "Interactive Gait Selector preference")



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


def get_subject_weight() -> float | None:
    weight = model.get_local("subject_weight")
    if weight is not None:
        try:
            val = float(weight)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None
    return None


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

