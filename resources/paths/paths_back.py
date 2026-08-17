import os
import json
import json.decoder
from copy import deepcopy
from resources.custom_exceptions import *
import logging

# todo: find a way to match selected trcs with the mots?

"""
Basic paths management

Using the local save file "local.json" and a virtual save, those methods load, save and manipulate the paths 
required by the pipeline.

Direct calls to this file should be avoided, as it is advised to pass through "paths_access.py" or "paths_gui.py" 
instead.
"""

# logs for debug
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), "path_navigation.log"), level=logging.INFO)


# manipulate save file local.json:

def _load_json() -> dict[str, str | list[str]]:
    """Load data from "local.json" save file.

    Returns:
        dictionary of the json data.

    Raises:
        OSError: if failure to open .json file.
    """
    try:
        with open(os.path.join(os.path.dirname(__file__), "local.json")) as json_loc:
            logging.info("Successfully read save file \"local.json\" into virtual save.")
            return json.load(json_loc)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        logging.info("No save in file \"local.json\" at the moment. Creating new empty virtual save.")
        return {}


_LOCAL = _load_json()


def save_to_json() -> None:
    """Overwrites save file "local.json" with local virtual save. Create the save file if missing.

    Returns:
        None
    """
    with open(os.path.join(os.path.dirname(__file__), "local.json"), mode="w+") as json_file:
        json.dump(_LOCAL, json_file, indent=3)
        logging.info(f"Successfully pushed virtual save to file save.")
    _LOCAL.update(_load_json())


# manipulate virtual save _LOCAL:

def _update_local(key: str, value: str | list[str]) -> None:
    """Update value of the local save.

    Args:
        key: key to update. Create it if not already in local save.
        value: new value of the given key

    Returns:
        None
    """
    global _LOCAL
    _LOCAL[key] = value
    logging.info(f"Updated virtual save for key: \"{key}\" with value \"{value}\".")


def _remove_from_local(key: str) -> None:
    """Remove value of the local save.

    Args:
        key: key to update.

    Returns:
        None
    """
    global _LOCAL
    try:
        _LOCAL.pop(key)
        logging.info(f"Updated virtual save, removed value of key: \"{key}\".")
    except KeyError:
        logging.warning(f"Could not update virtual save, no key: \"{key}\" to remove.")
        pass


# test if all the files required to run the pipeline has been filled and save:


def _delete_invalid_paths():
    """Delete the invalid paths in the local save file.

    Returns:
        None
    """
    global _LOCAL

    faulty = []
    for key in list(_LOCAL.keys()):
        content = _LOCAL[key]

        # Skip non-path entries like subject_weight, use_offset_corrector, postprocessing_version, use_interactive_gait_selector, event_detection_mode
        if key in ["subject_weight", "use_offset_corrector", "postprocessing_version", "use_interactive_gait_selector", "event_detection_mode"] or not isinstance(content, (str, list)):
            continue

        # if we're checking a list, test each element of the list:
        if isinstance(content, list):
            # remove each inexisting elelment from the list:
            for element in content:
                if isinstance(element, str) and not os.path.exists(element):
                    content.remove(element)
                    faulty.append(element)
            # remove the list if empty:
            if len(content) == 0:
                _remove_from_local(key)

        # check singular element:
        elif isinstance(content, str) and not os.path.exists(content):
            _remove_from_local(key)
            faulty.append(content)

    if len(faulty) > 0:
        logging.warning(f"Some of the previous paths do not exist anymore and have been removed from local save: "
                        f"{faulty}.")
        save_to_json()
    else:
        logging.info(f"All previous paths still exists.")


_delete_invalid_paths()


def are_loadbearing_paths_filled(detail: bool = False) -> bool | tuple[bool, list[str]]:
    """Check if all required paths had been filled.

    Args:
        detail: whether to return the missing paths

    Returns:
        bool: whether all the needed paths are filled
        list[str]: if detail is True, list of the missing paths
    """

    if not detail:
        return get_local("output_path") is not None and ( (get_local("raw_directory") is not None) or (get_local("raw_mot") is not None and get_local("raw_trc") is not None) )

    else:
        res = True
        faulty = []
        if get_local("output_path") is None:
            res = False
            faulty.append("output_path")
        if get_local("raw_directory") is None:
            faulty.append("raw_directory")
            for name in ["raw_mot", "raw_trc"]:
                local_content = get_local(name)
                if local_content is None:
                    res = False
                    faulty.append(name)
        if res:
            logging.info(f"All load bearing paths of the save file are filled.")
        else:
            logging.warning(f"Missing load bearing paths: {faulty}")
        return res, faulty


# access path in local virtual save:


def get_local(key: str = None) -> str | list[str] | dict[str, str | list[str]] | None:
    """Returns data from virtual save. If no key is given, return a copy of the virtual save.

    Args:
        key: element to fetch in local paths. Optional.

    Returns:
        Local paths. Can be a string (lone path), a list of string (multiple paths)
        or a dictionary (full copy of local paths).
    """
    if key is not None:
        return deepcopy(_LOCAL[key]) if key in _LOCAL else None
    return deepcopy(_LOCAL)


def get_default_searching_path() -> str:
    """Return the directory to use as initial directory when selecting paths.

    Returns:
        str: base path to search in, output path if defined, else the Document folder.
    """
    loc = get_local("output_path")
    return loc if loc is not None else os.path.expanduser("~/Documents")


# setup paths for the virtual save (may require a save in local file).


def set_output_directory(selection: str | None) -> bool:
    """Set up the output path into the virtual save if valid.

    Args:
        selection: path to save as output directory if valid

    Returns:
        bool, whether the given path is a valid output directory
    """
    error_message = f"Attempt at updating output directory failed: "

    # case : no selection
    if (selection is None) or (not selection):
        error_message = error_message + f"no path selected."
        logging.warning(error_message)
        return False

    # case : not existing (I'm not even sure this is a possible case with current calls)
    if not os.path.exists(selection):
        error_message = error_message + f" selected path {selection} does not exist."
        logging.warning(error_message)
        return False

    # case: not writeable
    if not os.access(selection, os.W_OK):
        error_message = error_message + f"selected path {selection} is not writeable."
        logging.warning(error_message)
        return False

    _update_local("output_path", selection)
    return True


def set_raw_mots(selection: list[str] | None) -> (bool, str | None):
    """Set up the given MOT files into the virtual save if valid.

    Args:
        selection: filepaths to save if valid

    Returns:
        bool, whether there are valid files to process in the given list
        str | None, details such as invalid files or error message
    """
    error_message = "Failed attempt at updating selection of raw MOT files to process: "

    # case: no file selected
    if len(selection) == 0:
        error_message = error_message + f"no file selected."
        logging.warning(error_message)
        return False, error_message

    # checking validity of selected files
    faulty = []
    for file in selection:
        if not os.path.isfile(file):
            faulty.append(file)
            selection.remove(file)

    # case: no valid file left
    if len(selection) == 0:
        error_message = error_message + f"none of the selected files {faulty} were valid."
        logging.warning(error_message)
        return False, error_message

    # keeping trace of eventual invalid files
    if len(faulty) > 0:
        message = f"Selected files {faulty} are invalid MOT files to process and will not be processed."
        logging.info(message)
    else:
        message = None

    _update_local("raw_mot", selection)
    return True, message


def set_raw_trcs(selection: list[str] | None) -> (bool, str | None):
    """Set up the given TRC files into the virtual save if valid.

    Args:
        selection: filepaths to save if valid

    Returns:
        bool, whether there are valid files to process in the given list
        str | None, details such as invalid files or error message
    """
    error_message = "Failed attempt at updating selection of raw TRC files to process: "

    # case: no file selected
    if len(selection) == 0:
        error_message = error_message + f"no file selected."
        logging.warning(error_message)
        return False, error_message

    # checking validity of selected files
    faulty = []
    for file in selection:
        if not os.path.isfile(file):
            faulty.append(file)
            selection.remove(file)

    # case: no valid file left
    if len(selection) == 0:
        error_message = error_message + f"none of the selected files {faulty} were valid."
        logging.warning(error_message)
        return False, error_message

    # keeping trace of eventual invalid files
    if len(faulty) > 0:
        message = f"Selected files {faulty} are invalid TRC files to process and will not be processed."
        logging.info(message)
    else:
        message = None

    _update_local("raw_trc", selection)
    return True, message


def set_raw_directory(selection: str | None) -> (bool, str | None):
    """Set up the given directory into the virtual save if valid.

    Args:
        selection: filepath to save if valid

    Returns:
        bool, whether the given path is valid
        str | None, details such as the listed invalid files, or an error message
    """
    error_message = "Failed attempt at updating selection of directory whose files to process: "

    # case: no file selected
    if selection is None or not selection:
        error_message = error_message + f"no directory selected."
        logging.warning(error_message)
        return False, error_message

    if not os.path.isdir(selection):
        error_message = error_message + f"selected path {selection} is not a directory."
        logging.warning(error_message)
        return False, error_message

    _update_local("raw_directory", selection)
    return True, None


def set_osim_path(selection: str | None) -> (bool, str | None):
    """If valid, save selected OpenSim source folder in both local and file save.

    Returns:
        bool, whether the selection was valid.
        str, if selection invalid, description fo the issue.
    """
    message = "Osim source folder could not be updated: "
    if not os.path.isdir(selection):
        message = message + "Selected path is not a directory."
        logging.warning(message)
        return False, message
    if os.path.basename(selection) != "bin":
        message = message + "Selected directory is not \"bin\"."
        logging.warning(message)
        return False, message
    _update_local("osim_path", selection)
    save_to_json()
    return True, None


def set_scaled_model(selection: str | None) -> (bool, str | None):
    """If valid, save selected OpenSim model in both local and file save.

    Returns:
        bool, whether the selection was valid.
        str, if selection invalid, description fo the issue.
    """
    message = "Scaled Osim model could not be updated: "
    if selection is None or not selection:
        message = message + "No path selected."
        logging.warning(message)
        return False, message
    if not os.path.isfile(selection):
        message = message + "Selected path is not a file."
        logging.warning(message)
        return False, message
    _update_local("osim_scaled_model", selection)
    save_to_json()
    return True, None


def set_subject_weight(weight: float | str | None) -> (bool, str | None):
    """Save subject/participant weight (in kg) into virtual and local save.

    Args:
        weight: float or str representing body weight in kg.

    Returns:
        bool, whether the weight value is valid (>0).
        str | None, description if invalid.
    """
    if weight is None or weight == "":
        _remove_from_local("subject_weight")
        save_to_json()
        return True, None
    try:
        val = float(weight)
        if val <= 0:
            return False, "Subject weight must be greater than 0 kg."
        _update_local("subject_weight", val)
        save_to_json()
        return True, None
    except (ValueError, TypeError):
        return False, "Invalid weight value. Please enter a valid number."


# delete paths from the virtual save (require a save in local file).

def delete_output_directory() -> None:
    """Delete the selected output path from the virtual save.

    Returns:
        None
    """
    _remove_from_local("output_path")


def delete_raw_mot() -> None:
    """Delete the selected raw mots from the virtual save.

    Returns:
        None
    """
    _remove_from_local("raw_mot")


def delete_raw_trc() -> None:
    """Delete the selected raw trcs from the virtual save.

    Returns:
        None
    """
    _remove_from_local("raw_trc")

def delete_raw_directory() -> None:
    """Delete the selected raw directory from the virtual save.
    Returns: None
    """
    _remove_from_local("raw_directory")


def set_use_offset_corrector(value: bool) -> None:
    """Set the preference for treadmill offset corrector.

    Args:
        value: True to enable, False to disable.
    """
    _update_local("use_offset_corrector", bool(value))


def get_use_offset_corrector() -> bool:
    """Get the preference for treadmill offset corrector.

    Returns:
        bool: True if enabled or unset (default True), False if disabled.
    """
    val = get_local("use_offset_corrector")
    if val is not None:
        return bool(val)
    return True


def set_postprocessing_version(version: str) -> None:
    _update_local("postprocessing_version", str(version).lower())


def get_postprocessing_version() -> str:
    val = get_local("postprocessing_version")
    if val in ["v1", "v2"]:
        return str(val).lower()
    return "v2"


def set_use_interactive_gait_selector(value: bool) -> None:
    _update_local("use_interactive_gait_selector", bool(value))


def get_use_interactive_gait_selector() -> bool:
    val = get_local("use_interactive_gait_selector")
    if val is not None:
        return bool(val)
    return False


def set_event_detection_mode(mode: str) -> None:
    """Save the event detection mode preference.

    Args:
        mode: one of 'grf_v1', 'grf_v2', or 'hybrid'.
    """
    _update_local("event_detection_mode", str(mode).lower())


def get_event_detection_mode() -> str:
    """Get the saved event detection mode preference.

    Returns:
        str: 'grf_v1', 'grf_v2', or 'hybrid'. Defaults to 'hybrid'.
    """
    val = get_local("event_detection_mode")
    if val in ["grf_v1", "grf_v2", "hybrid"]:
        return str(val).lower()
    return "hybrid"  # default to hybrid
