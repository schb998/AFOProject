import os

from resources.paths.paths_back import get_local, MissingLoadbearingPathException
from resources.paths.paths_gui import missing_loadbearing_path


def _call_to_gui(reason: str) -> None:
    missing_loadbearing_path(reason)


def get_base_path() -> str:
    content = get_local("output_path")
    if content is not None:
        return content
    _call_to_gui("Please fill in missing output path.")
    content = get_local("output_path")
    if content is not None:
        return content
    raise MissingLoadbearingPathException("Output directory")


def get_scaled_model_file() -> str:
    content = get_local('osim_scaled_model')
    if content is not None:
        return content
    _call_to_gui("Please select scaled opensim model.")
    content = get_local('osim_scaled_model')
    if content is not None:
        return content
    raise MissingLoadbearingPathException("Scaled OpenSim model")


def get_base_model_file() -> str:
    content = get_local('osim_base_model')
    if content is not None:
        return content
    raise OSError("No scaled model in .local.json.")


def get_raw_mot_path() -> str | list[str]:
    content = get_local('raw_mot')
    return content if content is not None else os.path.join(get_local("output_path"), 'raw')


def get_corrected_mot_path() -> str:
    content = get_local('corrected_mot')
    return content if content is not None else os.path.join(get_local("output_path"), 'corrected_mot')


def get_segmented_mot_path() -> str:
    content = get_local('segmented_mot')
    path = content if content is not None else os.path.join(get_local("output_path"), 'segmented')
    os.makedirs(path, exist_ok=True)
    return path


def get_raw_trc_path() -> str:
    content = get_local('raw_trc')
    return content if content is not None else os.path.join(get_local("output_path"), 'raw')


def get_segmented_trc_path() -> str:
    content = get_local('segmented_trc')
    path = content if content is not None else os.path.join(get_local("output_path"), 'segmented')
    os.makedirs(path, exist_ok=True)
    return path


def get_external_loads_path() -> str:
    content = get_local('external_loads')
    path = content if content is not None else os.path.join(get_local("output_path"), 'external_loads')
    os.makedirs(path, exist_ok=True)
    return path


def get_ik_results_path() -> str:
    content = get_local('ik_results')
    path = content if content is not None else os.path.join(get_local("output_path"), 'ik_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_id_results_path() -> str:
    content = get_local('id_results')
    path = content if content is not None else os.path.join(get_local("output_path"), 'id_results')
    os.makedirs(path, exist_ok=True)
    return path


def get_power_filtered_path() -> str:
    content = get_local('power_filtered')
    path = content if content is not None else os.path.join(get_local("output_path"), 'power_filtered')
    os.makedirs(path, exist_ok=True)
    return path
