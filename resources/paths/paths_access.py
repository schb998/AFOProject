import os
from resources.custom_exceptions import *
import resources.paths.paths_back as model
from resources.paths.paths_gui import main as main_gui, quick_setup


def call_to_gui() -> None:
    main_gui()


def call_quick_setup() -> bool:
    return quick_setup()


def get_output_path() -> str:
    """Get the pipeline's saved output path.

    Returns:
        str, stored output path

    Raises:
        MissingPathException if no such path has been given

    """
    content = model.get_local("output_path")
    if content is None:
        raise MissingPathException("Output directory")
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


def get_corrected_mot_path() -> str:
    path = os.path.join(model.get_local("output_path"), 'corrected_mot')
    os.makedirs(path, exist_ok=True)
    return path


def get_segmented_path() -> str:
    path = os.path.join(model.get_local("output_path"), 'segmented')
    os.makedirs(path, exist_ok=True)
    return path


def get_external_loads_path() -> str:
    path = os.path.join(model.get_local("output_path"), 'external_loads')
    os.makedirs(path, exist_ok=True)
    return path


def get_ik_results_path() -> str:
    path = os.path.join(model.get_local("output_path"), 'ik_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_id_results_path() -> str:
    path = os.path.join(model.get_local("output_path"), 'id_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_power_filtered_path() -> str:
    path = os.path.join(model.get_local("output_path"), 'power_filtered')
    os.makedirs(path, exist_ok=True)
    return path

