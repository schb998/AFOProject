from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilenames
from tkinter.ttk import *
import resources.tkinter_toolbox as tbox
import resources.paths.paths_back as back

LABELS: dict[Label, str] = {}
BUTTON: Button
CURRENT_ROW = 0
default_title = "AFO project"

def _update_labels() -> None:
    """Update content labels of the gui.

    Returns:
        None
    """
    for lab in LABELS:
        lab.config(text=tbox.reformat(back.get_local(LABELS[lab])))


def _update_row() -> None:
    """Update current row number.

    Returns:
        None
    """
    global CURRENT_ROW
    CURRENT_ROW = CURRENT_ROW + 1


def _update_button() -> None:
    """Update button state depending on the validity of the paths required by the pipeline.

    Returns:
        None
    """
    global BUTTON
    if BUTTON is not None:
        result = back.are_loadbearing_paths_filled()
        if result is True:
            BUTTON.state(['!disabled'])
        else:
            BUTTON.state(['disabled'])


def _setup_output_directory(root: Tk) -> (Label, Label, Button, Button):
    """Set up a line on given window to manage output directory.

    Args:
        root: root Tk window

    Returns:
        Explanatory Tkinter Label of the path asked.
        Tkinter Label containing current content of the local save.
        Selection Tkinter Button.
        Deselection Tkinter Button.
    """
    label = Label(root, text="Output directory:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=tbox.reformat(back.get_local("output_path")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "output_path"})

    def select_button_click():
        message = "Select the directory in which to save the pipeline's files."
        tbox.infobox(message)
        value = askdirectory(title=default_title + message, initialdir=back.get_default_searching_path())
        valid = back.set_output_directory(value)
        if not valid:
            tbox.infobox("Selection does not match requirement. Issue could be existence or writeability.")
        _update_labels()
        _update_button()

    select_button = Button(root, text="Select",
                           command=lambda: {select_button_click()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    def unselect_button_click():
        back.delete_output_directory()
        _update_labels()
        _update_button()

    unselect_button = Button(root, text="Unselect",
                             command=lambda: {unselect_button_click()})
    unselect_button.grid(row=CURRENT_ROW, column=3, sticky=NW, pady=10)

    _update_row()
    return label, selected, select_button, unselect_button


def _set_up_raw_directory(root: Tk) -> (Label, Label, Button, Button):
    """Set up a line on given window to manage raw MOT files to process.

        Args:
            root: root Tk window

        Returns:
            Explanatory Tkinter Label of the path asked.
            Tkinter Label containing current content of the local save.
            Selection Tkinter Button.
            Deselection Tkinter Button.
        """
    label = Label(root, text="Directory to process:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=tbox.reformat(back.get_local("raw_directory")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_directory"})

    def select_button_click():
        message = "Select the directory whose files should be processed."
        tbox.infobox(message)
        selection = askdirectory(title=message, initialdir=back.get_default_searching_path())
        valid, detail = back.set_raw_directory(selection)
        if not valid:
            message = f"Selection does not match requirement."
            if detail is not None:
                message = message + " " + detail
            tbox.infobox(message)
            return
        if detail is not None:
            tbox.infobox(detail)
        _update_labels()
        _update_button()

    select_button = Button(root, text="Select",
                           command=lambda: {select_button_click()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    def unselect_button_click():
        back.delete_raw_directory()
        _update_labels()
        _update_button()

    unselect_button = Button(root, text="Unselect",
                             command=lambda: {unselect_button_click()})
    unselect_button.grid(row=CURRENT_ROW, column=3, sticky=NW, pady=10)

    _update_row()
    return label, selected, select_button, unselect_button


def _setup_raw_c3d(root: Tk) -> (Label, Label, Button, Button):
    """Set up a line on given window to manage raw C3D files to process.

    Args:
        root: root Tk window

    Returns:
        Explanatory Tkinter Label of the path asked.
        Tkinter Label containing current content of the local save.
        Selection Tkinter Button.
        Deselection Tkinter Button.
    """
    label = Label(root, text="Raw C3D files to process:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=tbox.reformat(back.get_local("raw_c3d")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_c3d"})

    def select_button_click():
        message = "Select the raw C3D files to process."
        tbox.infobox(message)
        selection = tbox.get_c3d_files(message, back.get_default_searching_path())
        selection.sort()
        valid, detail = back.set_raw_c3ds(selection)
        if not valid:
            message = f"Selection does not match requirement."
            if detail is not None:
                message = message + " " + detail
            tbox.infobox(message)
            return
        if detail is not None:
            tbox.infobox(detail)
        _update_labels()
        _update_button()

    select_button = Button(root, text="Select",
                           command=lambda: {select_button_click()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    def unselect_button_click():
        back.delete_raw_c3d()
        _update_labels()
        _update_button()

    unselect_button = Button(root, text="Unselect",
                             command=lambda: {unselect_button_click()})
    unselect_button.grid(row=CURRENT_ROW, column=3, sticky=NW, pady=10)

    _update_row()
    return label, selected, select_button, unselect_button


def _setup_raw_mot(root: Tk) -> (Label, Label, Button, Button):
    """Set up a line on given window to manage raw MOT files to process.

    Args:
        root: root Tk window

    Returns:
        Explanatory Tkinter Label of the path asked.
        Tkinter Label containing current content of the local save.
        Selection Tkinter Button.
        Deselection Tkinter Button.
    """
    label = Label(root, text="Raw MOT files to process:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=tbox.reformat(back.get_local("raw_mot")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_mot"})

    def select_button_click():
        message = "Select the raw MOT files to process."
        tbox.infobox(message)
        selection = tbox.get_mot_files(message, back.get_default_searching_path())
        selection.sort()
        valid, detail = back.set_raw_mots(selection)
        if not valid:
            message = f"Selection does not match requirement."
            if detail is not None:
                message = message + " " + detail
            tbox.infobox(message)
            return
        if detail is not None:
            tbox.infobox(detail)
        _update_labels()
        _update_button()

    select_button = Button(root, text="Select",
                           command=lambda: {select_button_click()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    def unselect_button_click():
        back.delete_raw_mot()
        _update_labels()
        _update_button()

    unselect_button = Button(root, text="Unselect",
                             command=lambda: {unselect_button_click()})
    unselect_button.grid(row=CURRENT_ROW, column=3, sticky=NW, pady=10)

    _update_row()
    return label, selected, select_button, unselect_button


def _setup_raw_trc(root: Tk) -> (Label, Label, Button, Button):
    """Set up a line on given window to manage raw TRC files to process.

    Args:
        root: root Tk window

    Returns:
        Explanatory Tkinter Label of the path asked.
        Tkinter Label containing current content of the local save.
        Selection Tkinter Button.
        Deselection Tkinter Button.
    """
    label = Label(root, text="Raw TRC files to process:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=tbox.reformat(back.get_local("raw_trc")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_trc"})

    def select_button_click():
        message = "Select the raw TRC files to process."
        tbox.infobox(message)
        selection = tbox.get_trc_files(message, back.get_default_searching_path())
        selection.sort()
        valid, detail = back.set_raw_trcs(selection)
        if not valid:
            message = f"Selection does not match requirement."
            if detail is not None:
                message = message + " " + detail
            tbox.infobox(message)
            return
        if detail is not None:
            tbox.infobox(detail)
        _update_labels()
        _update_button()

    select_button = Button(root, text="Select",
                           command=lambda: {select_button_click()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    def unselect_button_click():
        back.delete_raw_trc()
        _update_labels()
        _update_button()

    unselect_button = Button(root, text="Unselect",
                             command=lambda: {unselect_button_click()})
    unselect_button.grid(row=CURRENT_ROW, column=3, sticky=NW, pady=10)

    _update_row()
    return label, selected, select_button, unselect_button


def _setup_or_label(root: Tk) -> Label:
    """Set up a quick label containing the word "or".

    Args:
        root: Tkinter root window

    Returns:
        the label

    """
    or_label = Label(root, text="OR")
    or_label.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    _update_row()
    return or_label


def main() -> None:
    """GUI managing the save file and the different paths required by the pipeline.

    Returns:
        None

    """
    root = Tk()
    root.title("Path management")
    tbox.set_up_window(root, window_width=600, window_height=400)
    root.columnconfigure(4)

    _setup_output_directory(root)

    _set_up_raw_directory(root)
    _setup_or_label(root)
    _setup_raw_c3d(root)
    _setup_or_label(root)
    _setup_raw_mot(root)
    _setup_raw_trc(root)

    save_button = Button(root, text="Save to file", default='active',
                         command=lambda: {back.save_to_json()})
    save_button.grid(row=CURRENT_ROW, column=0)

    proceed_button = Button(root, text="Proceed", default='active',
                            command=lambda: {root.destroy()})
    proceed_button.grid(row=CURRENT_ROW, column=1)

    global BUTTON
    BUTTON = proceed_button
    _update_button()

    root.protocol("WM_DELETE_WINDOW", lambda: tbox.on_closing(root, "Unsaved change will be lost."))
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()


if __name__ == "__main__":
    main()
