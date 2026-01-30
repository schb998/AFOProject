from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
from tkinter.filedialog import askopenfile, askopenfilenames, askdirectory
from tkinter.messagebox import showinfo
import io
from enum import Enum

class FileTypes(Enum):
    OSIM = [('OpenSim Project files', '*.osim')]
    TRC = [('OpenSim Marker files', '*.trc')]
    MOT = [('OpenSim Motion files', '*.mot')]
    XML = [('Extensible Markup Language files', '*.xml')]
    C3D = [('Motion Capture file', '*.c3d')]


def reformat(string: list[str] | tuple[str] | str | None) -> str:
    """Reformat given object into a comprehensible string for GUI usage.

    Args:
        string: object to reformat

    Returns:
        str: reformatted string
    """
    if string is None:
        return "empty"
    if isinstance(string, str):
        return string
    length = len(string)
    s = ""
    for i in range(length):
        s = s + string[i] + "\n" if i != length - 1 else s + string[i]
    return s


def set_up_window(window: Tk, window_width: int = 500, window_height: int = 300) -> None:
    """Sets up a Tk window with given width and weight.

    Args:
        window: Tk window
        window_width: int, width of the window
        window_height: int, height of the window

    Returns:
        None

    """
    # get the screen dimension
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')


def infobox(message: str, title=None) -> None:
    """Displays a tkinter information window with given message.

    Args:
        title:
        message: message to display

    Returns:
        None
    """
    showinfo(title=title if title is not None else "Information", message=message)


def on_closing(window: Tk, custom: str = None) -> bool:
    """Handle user closing given window.

    Args:
        window: window the user wants to close
        custom: custom message to display. Optional.

    Returns:
        bool, whether the user confirmed closing.
    """
    if custom is None:
        custom = "Do you want to quit?"
    if messagebox.askokcancel("Quit", custom):
        window.destroy()
        return True
    return False


def ask_question(question: str, title: str = None) -> bool:
    """Ask a questions to the user

    Args:
        question: str, question to display
        title: str, title of the window, optional

    Returns:

    """
    answer = messagebox.askquestion(title, question)
    return answer.lower() == "yes"


def _window_title_management(title: str = None, instruction: str = None, backup: str = "Selection window") -> str:
    """manage and standardize the titles of file selection windows.

    Args:
        title: str, title of the window
        instruction: str, additioanl instructiosn to give to the user
        backup: str, backup title in case of issue

    Returns:
        str, title of the window

    """
    full_title = ""
    if title is not None:
        full_title = full_title + title
    if instruction is not None:
        if not full_title:
            full_title = instruction
        else:
            full_title = full_title + " - " + instruction
    if not full_title:
        full_title = backup
    return full_title


def get_osim_file(instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    """Opens a window input requiring an OSIM file.

    Args:
        instruction: str, instruction to give to the user
        mode: str, access mode of the file to select, 'r' by default
        title: str, title of the window

    Returns:
        io.TextIOWrapper | None, selected file
    """
    return askopenfile(title=_window_title_management(title, instruction, "Select OSIM file."),
                       mode=mode,
                       filetypes=FileTypes.OSIM.value)


def get_osim_files(instruction: str = None, initialdir: str = None, title=None) -> list[str] | None:
    """Opens a window input requiring OSIM files.

    Args:
        instruction: str, instruction to give to the user
        initialdir: str, default directory to search in
        title: str, title of the window

    Returns:
        list of string, selected files
    """
    return list(askopenfilenames(title=_window_title_management(title, instruction, "Select OSIM files."),
                                 initialdir=initialdir,
                                 filetypes=FileTypes.OSIM.value))


def get_mot_file(instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    """Opens a window input requiring an MOT file.

    Args:
        instruction: str, instruction to give to the user
        mode: str, access mode of the file to select, 'r' by default
        title: str, title of the window

    Returns:
        io.TextIOWrapper | None, selected file
    """
    return askopenfile(title=_window_title_management(title, instruction, "Select MOT file."),
                       mode=mode,
                       filetypes=FileTypes.MOT.value)


def get_mot_files(instruction: str = None, initialdir: str = None, title=None) -> list[str] | None:
    """Opens a window input requiring MOT files.

    Args:
        instruction: str, instruction to give to the user
        initialdir: str, default directory to search in
        title: str, title of the window

    Returns:
        list of string, selected files
    """
    return list(askopenfilenames(title=_window_title_management(title, instruction, "Select MOT files."),
                                 initialdir=initialdir,
                                 filetypes=FileTypes.MOT.value))


def get_xml_file(instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    """Opens a window input requiring a XML file.

    Args:
        instruction: str, instruction to give to the user
        mode: str, access mode of the file to select, 'r' by default
        title: str, title of the window

    Returns:
        io.TextIOWrapper | None, selected file
    """
    return askopenfile(title=_window_title_management(title, instruction, "Select XML file."),
                       mode=mode,
                       filetypes=FileTypes.XML.value)


def get_xml_files(instruction: str = None, initialdir: str = None, title=None) -> list[str] | None:
    """Opens a window input requiring XML files.

    Args:
        instruction: str, instruction to give to the user
        initialdir: str, default directory to search in
        title: str, title of the window

    Returns:
        list of string, selected files
    """
    return list(askopenfilenames(title=_window_title_management(title, instruction, "Select XML files."),
                                 initialdir=initialdir,
                                 filetypes=FileTypes.XML.value))


def get_trc_file(instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    """Opens a window input requiring a TRC file.

    Args:
        instruction: str, instruction to give to the user
        mode: str, access mode of the file to select, 'r' by default
        title: str, title of the window

    Returns:
        io.TextIOWrapper | None, selected file
    """
    return askopenfile(title=_window_title_management(title, instruction, "Select TRC file."),
                       mode=mode,
                       filetypes=FileTypes.TRC.value)


def get_trc_files(instruction: str = None, initialdir: str = None, title=None) -> list[str] | None:
    """Opens a window input requiring TRC files.

    Args:
        instruction: str, instruction to give to the user
        initialdir: str, default directory to search in
        title: str, title of the window

    Returns:
        list of string, selected files
    """
    return list(askopenfilenames(title=_window_title_management(title, instruction, "Select TRC files."),
                                 initialdir=initialdir,
                                 filetypes=FileTypes.TRC.value))


def get_c3d_file(instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    """Opens a window input requiring a C3D file.

    Args:
        instruction: str, instruction to give to the user
        mode: str, access mode of the file to select, 'r' by default
        title: str, title of the window

    Returns:
        io.TextIOWrapper | None, selected file
    """
    return askopenfile(title=_window_title_management(title, instruction, "Select C3D file."),
                       mode=mode,
                       filetypes=FileTypes.C3D.value)


def get_c3d_files(instruction: str = None, initialdir: str = None, title=None) -> list[str] | None:
    """Opens a window input requiring C3D files.

    Args:
        instruction: str, instruction to give to the user
        initialdir: str, default directory to search in
        title: str, title of the window

    Returns:
        list of string, selected files
    """
    res = askopenfilenames(title=_window_title_management(title, instruction, "Select C3D files."),
                                 initialdir=initialdir,
                                 filetypes=FileTypes.C3D.value)
    return list(res)


def get_file(types: list[FileTypes], instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    """Opens a window input requiring a file of given format.

    Args:
        types: list[FileTypes], possible formats for the researched file
        instruction: str, instruction to give to the user
        mode: str, access mode of the file to select, 'r' by default
        title: str, title of the window

    Returns:
        io.TextIOWrapper | None, selected file
    """
    filet = [t.value[0] for t in types]
    return askopenfile(title=_window_title_management(title, instruction, "Select a file."),
                       mode=mode,
                       filetypes=filet)