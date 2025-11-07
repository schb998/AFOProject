import opensim as osim
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile, askopenfilenames, askdirectory
import os
import sys
import TreadMetrix.wip_pipeline.tkinter_toolbox as tbox


def configure_opensim() -> None:
    """Locally configures Opensim.

    Returns:
        None
    """
    message = "Select OpenSim's source folder, \"bin\" directory"
    tbox.infobox(message)
    opensim_path = askdirectory(title=message)
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
        raise Exception("No base model selected. Interrupting.")
    base_model_filename = base_model_file.name
    base_model = osim.Model(base_model_filename)

    message = "Select the static file."
    tbox.infobox(message)
    static = tbox.get_trc_file(instruction=message)
    if static is None:
        raise Exception("No static file selected. Interrupting.")
    static_filename = static.name

    message = "Select the scaling setup file."
    tbox.infobox(message)
    scale_setup = tbox.get_xml_file(instruction=message)
    if scale_setup is None:
        raise Exception("No static file selected. Interrupting.")
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
    root.title("Model scaling and IK via Opensim")
    tbox.set_up_window(root)

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



