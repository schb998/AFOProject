from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile, askopenfilenames, askdirectory
from tkinter.messagebox import showinfo
import io


def set_up_window(window: Tk, window_width: int = 500, window_height: int = 300) -> None:
    # get the screen dimension
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')


def infobox(message: str) -> None:
    """Displays a tkinter information window with gicen message

    Args:
        message: message to display

    Returns:
        None
    """
    showinfo(title='Information', message=message)


def get_osim_file(instruction: str = None, mode: str = 'r') -> io.TextIOWrapper | None:
    return askopenfile(title=instruction, mode=mode, filetypes=[('OpenSim Project files', '*.osim')])


def get_mot_file(instruction: str = None, mode: str = 'r') -> io.TextIOWrapper | None:
    return askopenfile(title=instruction, mode=mode, filetypes=[('OpenSim Motion files', '*.mot')])


def get_xml_file(instruction: str = None, mode: str = 'r') -> io.TextIOWrapper | None:
    return askopenfile(title=instruction, mode=mode, filetypes=[('Extensible Markup Language files', '*.xml')])


def get_trc_file(instruction: str = None, mode: str = 'r') -> io.TextIOWrapper | None:
    return askopenfile(title=instruction, mode=mode, filetypes=[('OpenSim Marker files', '*.trc')])
