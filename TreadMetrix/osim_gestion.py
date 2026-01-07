import opensim as osim
from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter.ttk import *
import os
import sys
import resources.tkinter_toolbox as tbox
from resources.custom_exceptions import *
import resources.paths.paths_back as m
import resources.paths.paths_access as c

# todo: update configure_opensim so paths_back manage configuration instead
# todo: continue separation osim_gestion // paths_back for model selection

LABEL: Label
BUTTON: Button
CURRENT_ROW = 0

setup = r"C:\Users\lgre690\Documents\MyData\osim_code\setup_final.xml"
base_model = r"C:\Users\lgre690\Documents\MyData\osim_code\gait2392_simbody_37 ABI full markerset _ lilas.osim"
static = r"lilas\Static_0102.trc"
output = r"C:\Users\lgre690\Documents\MyData\osim_code\lilas\LALALALWEGOTARESULT.osim"

def _update_label():
    global LABEL
    if LABEL is not None:
        try:
            txt = c.get_osim_path()
            LABEL.config(text=txt)
        except MissingPathException:
            LABEL.config(text="empty")


def _update_button():
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
        _update_label()
        _update_button()

    select_button = Button(root, text="Select new", command=button_click)
    select_button.grid(row=0, column=2, sticky=NW, pady=10)

    confirm_button = Button(root, text="Confirm", command=root.destroy)
    confirm_button.grid(row=1, column=1, sticky=NW, pady=10)
    confirm_button.state(['disabled'])
    BUTTON = confirm_button

    _update_button()

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


def scale_model(base_model_filename:str = None, static_filename: str = None, scale_setup_filename: str = None):
    if base_model_filename is None:
        message = "Select the base model file."
        tbox.infobox(message)
        base_model_file = tbox.get_osim_file(instruction=message)
        if base_model_file is None:
            raise MissingPathException("OpenSim base project file", "interrupting")
        base_model_filename = base_model_file.name

    if static_filename is None:
        message = "Select the static file."
        tbox.infobox(message)
        static = tbox.get_trc_file(instruction=message)
        if static is None:
            raise MissingPathException("Static TRC file", "interrupting")
        static_filename = static.name

    if scale_setup_filename is None:
        message = "Select the scaling setup file."
        tbox.infobox(message)
        scale_setup = tbox.get_xml_file(instruction=message)
        if scale_setup is None:
            raise MissingPathException("Scaling tool setup XML file", "interrupting")
        scale_setup_filename = scale_setup.name

    scale_tool = osim.ScaleTool(scale_setup_filename)
    scale_tool.getModelScaler().setMarkerFileName(static_filename)

    scale_tool.setPrintResultFiles(True)
    scale_tool.getModelScaler().setOutputModelFileName(output)
    scale_tool.getMarkerPlacer().setOutputModelFileName(output)
    scale_tool.getMarkerPlacer().setCoordinateFileName(static_filename)

    scale_tool.run()

    print("Something happened")


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

    button1 = Button(root, text="Select scaled model", command=select_scaled_model)
    button2 = Button(root, text="Scale a model", command=scale_model)
    button1.pack(ipadx=5, ipady=5, expand=True)
    button2.pack(ipadx=5, ipady=5, expand=True)

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()


if __name__ == "__main__":
    # main()
    scale_model(base_model, static, setup)




