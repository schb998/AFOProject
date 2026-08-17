from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
from tkinter.filedialog import askopenfile, askopenfilenames, askdirectory
from tkinter.messagebox import showinfo
import io

def reformat(string: list[str] | tuple[str] | str | None) -> str:
    """Reformat given object into a comprehensible string for gui usage.

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
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    answer = messagebox.askquestion(title, question, parent=root)
    root.destroy()
    return answer.lower() == "yes"


def get_osim_file(instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    t = ""
    if title is not None:
        t = t + title
    if instruction is not None:
        if not t:
            t = instruction
        else:
            t = t + " - " + instruction
    if not t:
        t = "Select OSIM file."
    return askopenfile(title=t, mode=mode, filetypes=[('OpenSim Project files', '*.osim')])


def get_mot_file(instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    t = ""
    if title is not None:
        t = t + title
    if instruction is not None:
        if not t:
            t = instruction
        else:
            t = t + " - " + instruction
    if not t:
        t = "Select MOT file."
    return askopenfile(title=t, mode=mode, filetypes=[('OpenSim Motion files', '*.mot')])


def get_xml_file(instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    t = ""
    if title is not None:
        t = t + title
    if instruction is not None:
        if not t:
            t = instruction
        else:
            t = t + " - " + instruction
    if not t:
        t = "Select XML file."
    return askopenfile(title=t, mode=mode, filetypes=[('Extensible Markup Language files', '*.xml')])


def get_trc_file(instruction: str = None, mode: str = 'r', title=None) -> io.TextIOWrapper | None:
    t = ""
    if title is not None:
        t = t + title
    if instruction is not None:
        if not t:
            t = instruction
        else:
            t = t + " - " + instruction
    if not t:
        t = "Select TRC file."
    return askopenfile(title=t, mode=mode, filetypes=[('OpenSim Marker files', '*.trc')])
