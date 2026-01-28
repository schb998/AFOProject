import opensim as osim
import shutil
from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter.ttk import *
import os
import sys
import resources.tkinter_toolbox as tbox
from TreadMetrix.hip_joint_computation import compute_hip_joints
from resources.custom_exceptions import *
import resources.paths.paths_back as m
import resources.paths.paths_access as c
import xml.etree.ElementTree as ET

from resources.file_types.trc import TRC

# todo: update configure_opensim so paths_back manage configuration instead
# todo: continue separation osim_gestion // paths_back for model selection

LABEL: Label
BUTTON: Button
CURRENT_ROW = 0

_osim_files_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), r"resources/osim_files")
_setup = os.path.join(_osim_files_path, "scaling_setup.xml").replace("\\", "/")
_base_model = os.path.join(_osim_files_path, "markerset.osim").replace("\\", "/")


def _update_scaling_setup_file(scaling_setup: str, base_model_filepath: str = None) -> None:
    """Update the scaling setup file to have the valid absolute path of the base model file

    Args:
        scaling_setup: str, path to the scaling setup to update
        base_model_filepath: str, path to the base model file if the default one is not to be used.

    Returns:
        None

    """
    tree = ET.parse(scaling_setup, ET.XMLParser(encoding="utf-8"))
    model_file = tree.getroot().find("ScaleTool").find("GenericModelMaker").find('model_file')
    txt = model_file.text
    if base_model_filepath is None:
        base_model_filepath = _base_model
    if txt != base_model_filepath:
        for x in tree.getroot().find("ScaleTool").find("GenericModelMaker").iter('model_file'):
            x.text = str(base_model_filepath)
        tree.write(scaling_setup, encoding="utf-8", xml_declaration=True)


def _prepare_scaling_setup(directory_path: str, base_model_filepath: str = None) -> None:
    shutil.copy(_setup, directory_path)
    _update_scaling_setup_file(os.path.join(directory_path, "scaling_setup.xml"), base_model_filepath)


def _update_base_opensim_label():
    global LABEL
    if LABEL is not None:
        try:
            txt = c.get_osim_path()
            LABEL.config(text=txt)
        except MissingPathException:
            LABEL.config(text="empty")


def _update_base_opensim_setup_button():
    global BUTTON
    if BUTTON is not None:
        try:
            txt = c.get_osim_path()
            BUTTON.state(['!disabled'])
        except MissingPathException:
            BUTTON.state(['disabled'])


def configure_opensim() -> None:
    """Locally configures Opensim.

    Returns:
        None
    """
    global LABEL
    global BUTTON

    root = Tk()
    root.title("OpenSim's source folder")
    tbox.set_up_window(root, window_width=800)
    root.columnconfigure(2)

    label = Label(root, text="Saved path:")
    label.grid(row=0, column=0, sticky=NW, pady=10)

    def custom_on_closing(window: Tk):
        try:
            content = c.get_osim_path()
            tbox.on_closing(window, f"Proceed with path: \"{content}\"?")
        except MissingPathException:
            closed = tbox.on_closing(window, "No OpenSim path was given. Closing this window will interrupt processing.")
            if closed:
                raise KeyboardInterrupt


    root.protocol("WM_DELETE_WINDOW", lambda: custom_on_closing(root))

    try:
        txt = c.get_osim_path()
    except MissingPathException:
        txt = "empty"

    selected = Label(root, text=txt, background="darkgrey")
    selected.grid(row=0, column=1, sticky=NW, pady=10)
    LABEL = selected

    def button_click():
        message = "Select OpenSim's source folder, \"bin\" directory"
        tbox.infobox(message)
        selection = askdirectory(title=message,
                                 initialdir=os.path.expanduser("~/Documents"))
        success, reason = m.set_osim_path(selection)
        if not success:
            tbox.infobox(reason)
        _update_base_opensim_label()
        _update_base_opensim_setup_button()

    select_button = Button(root, text="Select new", command=button_click)
    select_button.grid(row=0, column=2, sticky=NW, pady=10)

    confirm_button = Button(root, text="Confirm", command=root.destroy)
    confirm_button.grid(row=1, column=1, sticky=NW, pady=10)
    confirm_button.state(['disabled'])
    BUTTON = confirm_button

    _update_base_opensim_setup_button()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()

    opensim_path = c.get_osim_path()

    os.environ['OPENSIM_HOME'] = opensim_path
    os.add_dll_directory(opensim_path)
    sys.path.append(os.path.join(opensim_path, 'Bindings', 'Python'))
    os.environ['PATH'] += os.pathsep + os.path.join(opensim_path, 'bin')


def select_scaled_model() -> None:
    message = "Please select scaled model file."
    tbox.infobox(message)
    file = tbox.get_osim_file(instruction=message)
    success, detail = m.set_scaled_model(file.name if file is not None else None)
    if not success:
        tbox.infobox(detail)
    _update_scaled_model_label()
    _update_scaled_model_button()


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


def scale_model(static_filepath: str = None, output_filepath: str = None, scaling_setup_file: str = None,
                base_model_file: str = None, keep_setup: bool = False) -> str | None:
    """Scale an Osim model according to the static file provided.

    Args:
        static_filepath: str, path to the static file (trc)
        output_filepath: str, path to the directory in which to save the scaled model.
            If none, scaled model will be saved in the same directory as the static file.
        scaling_setup_file: str, path to an XML scaling set up file if default setup is not to be used
        base_model_file: str, path to the base model file if default model is not to be used
        keep_setup: bool, whether to keep the scaling setup file

    Returns:
        Full path to the scaled model if scaling was a success, None if it failed.

    """

    while static_filepath is None:
        message = 'Select the static file.'
        tbox.infobox(message)
        static_filepath = tbox.get_file([tbox.FileTypes.TRC, tbox.FileTypes.C3D], instruction=message).name

    dirname = os.path.dirname(static_filepath)
    if static_filepath.endswith(TRC.extension):
        added_hj_filename = compute_hip_joints(static_filepath, output_path=dirname).filename
    else:
        trc = TRC.load_from_c3d(static_filepath)
        trc.save(dirname)
        static_filepath = trc.filepath
        added_hj_filename = compute_hip_joints(static_filepath, trc, output_path=dirname).filename


    output_file = output_filepath if output_filepath is not None else static_filepath.replace(".trc", "_scaled_model.osim")
    base_model_file = base_model_file if base_model_file is not None else _base_model

    if scaling_setup_file is None:
        scaling_directory = os.path.dirname(static_filepath)
        _prepare_scaling_setup(scaling_directory, base_model_file)
        scaling_setup_file = os.path.join(scaling_directory, "scaling_setup.xml").replace("\\", "/")

    scale_tool = osim.ScaleTool(scaling_setup_file)

    scale_tool.getModelScaler().setMarkerFileName(added_hj_filename)
    scale_tool.setPrintResultFiles(True)
    scale_tool.getModelScaler().setOutputModelFileName(output_file)
    scale_tool.getMarkerPlacer().setOutputModelFileName(output_file)
    scale_tool.getMarkerPlacer().setCoordinateFileName(added_hj_filename)

    scale_tool.run()

    success, detail = m.set_scaled_model(output_file)
    if not success:
        tbox.infobox(detail)
    if not keep_setup:
        os.remove(scaling_setup_file)

    return output_file if success else None


def setup_ik_tool() -> osim.InverseKinematicsTool | None:
    """Ask the user to select a setup file for OpenSim's IK tool (XML file) and return an InverseKinematicsTool object.

    Returns:
        An InverseKinematicsTool object if the selected file is valid, None if not.

    """
    ik_setup = tbox.get_xml_file('Select the XML set up file for IK tool.')
    if ik_setup is not None:
        return osim.InverseKinematicsTool(ik_setup)
    return None


def _update_scaled_model_label():
    global LABEL
    if LABEL is not None:
        try:
            txt = c.get_scaled_model_file()
            LABEL.config(text=tbox.reformat(txt))
        except MissingPathException:
            LABEL.config(text="empty")


def _update_scaled_model_button():
    global BUTTON
    if BUTTON is not None:
        try:
            c.get_scaled_model_file()
            BUTTON.state(['!disabled'])
        except MissingPathException:
            BUTTON.state(['disabled'])


def main() -> None:
    """Open the user interface to manage OpenSim-related set up.

    Returns:
        None

    """
    configure_opensim()

    root = Tk()
    root.title("Model scaling and IK via OpenSim")
    tbox.set_up_window(root, window_width=700)
    root.columnconfigure(3)

    scaling_button = Button(root, text="Scale a model", command=lambda: {scale_model(),
                                                                         _update_scaled_model_label(),
                                                                         _update_scaled_model_button()})
    selection_button = Button(root, text="Select scaled model", command=select_scaled_model)

    label = Label(root, text="Selected scaled model:")
    selected = Label(root, text=tbox.reformat(m.get_local("osim_scaled_model")), background="darkgrey")
    global LABEL
    LABEL = selected
    _update_scaled_model_label()

    proceed_button = Button(root, text="Proceed", default='active',
                            command=lambda: {root.destroy()})

    label.grid(row=0, column=0, sticky=NW, pady=10)
    selected.grid(row=0, column=1, sticky=NW, pady=10)
    scaling_button.grid(row=1, column=0, sticky=NE, pady=10)
    selection_button.grid(row=1, column=1, sticky=NW, pady=10)
    proceed_button.grid(row=1, column=2, sticky=NW, pady=10)

    global BUTTON
    BUTTON = proceed_button
    _update_scaled_model_button()



    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()


if __name__ == "__main__":
     main()
