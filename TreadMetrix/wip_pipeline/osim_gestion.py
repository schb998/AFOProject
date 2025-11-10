import opensim as osim
from tkinter import *
from tkinter.ttk import *
import os
import sys
import resources.tkinter_toolbox as tbox
from resources.custom_exceptions import *
from resources.paths.paths_back import set_osim_path, get_local
from resources.paths.paths_access import get_osim_path, get_base_model_file, get_scaled_model_file

LABELS: dict[Label, str] = {}
BUTTONS: dict[Button, str] = {}
CURRENT_ROW = 0


def _update_labels():
    for lab in LABELS:
        content = get_local(LABELS[lab])
        txt = content if content is not None else "empty"
        lab.config(text=txt)


def _update_buttons():
    for button in BUTTONS:
        content = get_local(BUTTONS[button])
        if content is None:
            button.state(['disabled'])
        else:
            button.state(['!disabled'])


def configure_opensim() -> None:
    """Locally configures Opensim.

    Returns:
        None
    """

    root = Tk()
    root.title("OpenSim's source folder")
    tbox.set_up_window(root, window_width=800)
    root.columnconfigure(2)

    label = Label(root, text="Saved path:")
    label.grid(row=0, column=0, sticky=NW, pady=10)


    selected = Label(root, text=get_osim_path())
    selected.grid(row=0, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "osim_path"})

    def button_click():
        set_osim_path()
        _update_labels()
        _update_buttons()

    select_button = Button(root, text="Select new", command=button_click)
    select_button.grid(row=1, column=0, sticky=NW, pady=10)

    confirm_button = Button(root, text="Confirm", command=root.destroy)
    confirm_button.grid(row=1, column=1, sticky=NW, pady=10)
    BUTTONS.update({confirm_button: "osim_path"})
    _update_buttons()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()

    opensim_path = get_osim_path()


    os.environ['OPENSIM_HOME'] = opensim_path
    os.add_dll_directory(opensim_path)
    sys.path.append(os.path.join(opensim_path, 'Bindings', 'Python'))
    os.environ['PATH'] += os.pathsep + os.path.join(opensim_path, 'bin')


def open_scaled_model() -> str | None:
    message = "Please select scaled model file."
    tbox.infobox(message)
    file = tbox.get_osim_file(instruction=message)
    if file is not None:
        return file.read()
    return None


def get_scaled_model_filename() -> str | None:
    message = "Please select scaled model file."
    tbox.infobox(message)
    file = tbox.get_osim_file(instruction=message)
    if file is not None:
        return file.name
    return None


def open_base_model_file() -> str | None:
    message = "Please select model file to be scaled."
    tbox.infobox(message)
    file = tbox.get_osim_file(instruction=message)
    if file is not None:
        return file.read()
    return None


def get_base_model_filename() -> str | None:
    message = "Please select model file to be scaled."
    tbox.infobox(message)
    file = tbox.get_osim_file(instruction=message)
    if file is not None:
        return file.name
    return None



def scale_model() -> str:
    message = "Select the base model file."
    tbox.infobox(message)
    base_model_file = tbox.get_osim_file(instruction=message)
    if base_model_file is None:
        raise MissingPathException("OpenSim base project file", "none", "interrupting")
    base_model_filename = base_model_file.name
    base_model = osim.Model(base_model_filename)

    message = "Select the static file."
    tbox.infobox(message)
    static = tbox.get_trc_file(instruction=message)
    if static is None:
        raise MissingPathException("Static TRC file", "none", "interrupting")
    static_filename = static.name

    message = "Select the scaling setup file."
    tbox.infobox(message)
    scale_setup = tbox.get_xml_file(instruction=message)
    if scale_setup is None:
        raise MissingPathException("Scaling tool setup XML file", "none", "interrupting")
    scale_setup_filename = scale_setup.name

    scale_tool = osim.ScaleTool(scale_setup_filename)
    return static_filename


def setup_ik():
    setup = tbox.get_xml_file('Select the XML set up file for IK tool.')
    if setup is not None:
        return osim.InverseKinematicsTool(setup)
    return None


def main() -> None:
    configure_opensim()

    root = Tk()
    root.title("Model scaling and IK via OpenSim")
    tbox.set_up_window(root)
    root.columnconfigure(2)

    button1 = Button(root, text="Select scaled model", command=open_scaled_model)
    button2 = Button(root, text="Scale a model", command=scale_model)
    button1.pack(ipadx=5, ipady=5, expand=True)
    button2.pack(ipadx=5, ipady=5, expand=True)

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()


if __name__ == "__main__":
    main()



