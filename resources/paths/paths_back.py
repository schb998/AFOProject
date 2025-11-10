from tkinter.filedialog import askopenfilenames, askdirectory
import os
import resources.tkinter_toolbox as tbox
import json
import json.decoder
from copy import deepcopy
from resources.custom_exceptions import *


def _load_json() -> dict[str, str | list[str]]:
    """Load data from ".local.json" file. If file is missing, create it.

    Returns:
        dictionary of the json data.

    Raises:
        OSError if failure to open .json file.
    """
    try:
        with open(os.path.join(os.path.dirname(__file__), ".local.json")) as json_loc:
            return json.load(json_loc)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        return {}


_LOCAL = _load_json()


def _update_local(key: str, value: str | list[str]) -> None:
    global _LOCAL
    _LOCAL[key] = value


def _remove_from_local(key: str) -> None:
    global _LOCAL
    try:
        _LOCAL.pop(key)
    except KeyError:
        pass


def get_local(key: str = None) -> str | list[str] | dict[str, str | list[str]]:
    """Returns datad from local paths. If no key is given, return a copy of the lcoal paths dictionary.

    Args:
        key: element to fetch in local paths. Optional.

    Returns:
        Local paths. Can be a string (lone path), a list of string (multiple paths)
        or a dictionary (full copy of local paths).

    """
    if key is not None:
        return deepcopy(_LOCAL[key]) if key in _LOCAL else None
    return deepcopy(_LOCAL)


def save_to_json() -> None:
    """Save any change in local paths to the ".local.json" file. Create it if missing.

    Returns:
        None
    """
    with open(os.path.join(os.path.dirname(__file__), ".local.json"), mode="w+") as json_file:
        json.dump(_LOCAL, json_file, indent=3)
    _LOCAL.update(_load_json())


def _test_writability(path: str) -> bool:
    return os.access(path, os.W_OK)


def _test_readability(path: str) -> bool:
    return os.access(path, os.R_OK)


def _get_default_searching_path() -> str:
    return _LOCAL["output_path"] if "output_path" in _LOCAL else os.path.expanduser("~/Documents")


def set_output_directory() -> None:
    message = "Select the directory in which to save the pipeline's files."
    tbox.infobox(message)
    answer = askdirectory(title=message, initialdir=_get_default_searching_path())
    if not answer:
        tbox.infobox("No directory selected.")
        return
    if not _test_writability(answer):
        tbox.infobox("Selected directory is not writeable.")
        return
    _update_local("output_path", answer)


def delete_output_directory() -> None:
    _remove_from_local("output_path")


def set_raw_mots() -> None:
    message = "Select the raw MOT files to process."
    tbox.infobox(message)
    answer = askopenfilenames(title=message,
                              initialdir=_get_default_searching_path(),
                              filetypes=[("OpenSim Motion files", "*.mot")])
    if len(answer) == 0:
        tbox.infobox("No files selected.")
        return
    answer = list(answer)
    answer.sort()
    _update_local("raw_mot", answer)
    tbox.infobox(f"Selected files: {answer}.")


def delete_raw_mot() -> None:
    _remove_from_local("raw_mot")


def set_raw_trcs() -> None:
    # todo: find a way to manually match trcs with mots?
    message = "Select the raw TRC files to process."
    tbox.infobox(message)
    answer = askopenfilenames(title=message,
                              initialdir=_get_default_searching_path(),
                              filetypes=[("OpenSim Marker files", "*.trc")])
    if len(answer) == 0:
        tbox.infobox("No files selected.")
        return
    answer = list(answer)
    answer.sort()
    _update_local("raw_trc", answer)
    tbox.infobox(f"Selected files: {answer}.")


def delete_raw_trc() -> None:
    _remove_from_local("raw_trc")


def set_osim_path() -> None:
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
