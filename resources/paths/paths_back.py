from tkinter.filedialog import askopenfilenames, askdirectory
import os
import resources.tkinter_toolbox as tbox
import json
import json.decoder
from copy import deepcopy
from resources.custom_exceptions import *
import logging

# todo: clean up the gui/back interractions by sending all tkinter usages to gui
# todo: find a way to match selected trcs with the mots?

"""
Basic paths management

Using the local save file "local.json" and a virtual save, those methods load, save and manipulate the paths 
required by the pipeline.

Direct calls to this file should be avoided, as it is advised to pass through "paths_access.py" or "paths_gui.py" 
instead.
"""

# logs for debug
logger = logging.getLogger("test")
logging.basicConfig(filename='path_navigation.log', level=logging.INFO)


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

_load_bearing_paths: list[str] = ["output_path", "raw_mot", "raw_trc"]


def arevalid_loadbearing_paths(detail: bool = False) -> bool | tuple[bool, list[str]]:
    """Check if all required paths had been filled.

    Args:
        detail: whether to return the missing paths

    Returns:
        bool: whether all the needed paths are filled
        list[str]: if detail is True, list of the missing paths
    """
    global _load_bearing_paths

    if not detail:
        for name in _load_bearing_paths:
            local_content = get_local(name)
            if local_content is None:
                logging.warning(f"Missing load bearing paths.")
                return False
        logging.info(f"All load bearing paths of the save file are filled.")
        return True

    else:
        res = True
        problems = []
        for name in _load_bearing_paths:
            local_content = get_local(name)
            if local_content is None:
                res = False
                problems.append(name)
        if res:
            logging.info(f"All load bearing paths of the save file are filled.")
        else:
            logging.warning(f"Missing load bearing paths: {problems}")
        return res, problems


# access path in local virtual save:


def get_local(key: str = None) -> str | list[str] | dict[str, str | list[str]]:
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


def _get_default_searching_path() -> str:
    """Return the directory to use as initial directory when selecting paths.

    Returns:
        str: base path to search in, output path if defined, else the Document folder.
    """
    loc = get_local("output_path")
    return loc if loc is not None else os.path.expanduser("~/Documents")


# setup paths for the virtual save, may require a save in local file.


def set_output_directory() -> None:
    """Set up the output path from user input.

    Returns:
        None
    """
    message = "Select the directory in which to save the pipeline's files."
    tbox.infobox(message)
    answer = askdirectory(title=message, initialdir=_get_default_searching_path())
    if (not answer) or (not os.path.exists(answer)):
        tbox.infobox("No directory selected.")
        return
    if not os.access(answer, os.W_OK):
        tbox.infobox("Selected directory is not writeable.")
        return
    _update_local("output_path", answer)


def set_raw_mots() -> None:
    """Select the raw MOT files to process and save them in virtaul save.

    Returns:
        None
    """
    message = "Select the raw MOT files to process."
    tbox.infobox(message)
    selection = askopenfilenames(title=message,
                                 initialdir=_get_default_searching_path(),
                                 filetypes=[("OpenSim Motion files", "*.mot")])
    if len(selection) == 0:
        tbox.infobox("No files selected.")
        return
    selection = list(selection)
    selection.sort()
    tbox.infobox(f"Selected files: {selection}.")

    faulty = []
    for file in selection:
        if not os.path.isfile(file):
            faulty.append(file)
            selection.remove(file)

    if len(faulty) > 0:
        tbox.infobox(f"Selected files {faulty} are not valid and will not be processed.")
    _update_local("raw_mot", selection)


def set_raw_trcs() -> None:
    """Select the raw TRC files to process and save them in virtaul save.

    Returns:
        None
    """
    message = "Select the raw TRC files to process."
    tbox.infobox(message)
    selection = askopenfilenames(title=message,
                                 initialdir=_get_default_searching_path(),
                                 filetypes=[("OpenSim Marker files", "*.trc")])
    if len(selection) == 0:
        tbox.infobox("No files selected.")
        return
    selection = list(selection)
    selection.sort()
    tbox.infobox(f"Selected files: {selection}.")

    faulty = []
    for file in selection:
        if not os.path.isfile(file):
            faulty.append(file)
            selection.remove(file)

    if len(faulty) > 0:
        tbox.infobox(f"Selected files {faulty} are not valid and will not be processed.")
    _update_local("raw_trc", selection)



def set_osim_path() -> None:
    """Save selected OpenSim source folder in both local and file save.

    Returns:
        None

    Raises:
        WrongExtensionException: if selected path does not exist or is not a directory.
        InvalidPathException: if the selected directory is not the "bin" directory.
    """
    message = "Select OpenSim's source folder, \"bin\" directory"
    tbox.infobox(message)
    opensim_path = askdirectory(title=message,
                                initialdir=os.path.expanduser("~/Documents"))
    if not os.path.isdir(opensim_path):
        tbox.infobox("Selected path is not a directory.")
        raise WrongExtensionException("OpenSim's source folder", opensim_path)
    if os.path.basename(opensim_path) != "bin":
        tbox.infobox("Selected directory is not \"bin\".")
        raise InvalidPathException("OpenSim's source folder", opensim_path, "given path is not \"bin\" directory.")
    _update_local("osim_path", opensim_path)
    save_to_json()


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

