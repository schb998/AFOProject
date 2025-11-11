from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
import resources.tkinter_toolbox as tbox
import resources.paths.paths_back as back

LABELS: dict[Label, str] = {}
BUTTON: Button
CURRENT_ROW = 0


def _update_labels():
    """Update content labels of the gui.

    Returns:
        None
    """
    for lab in LABELS:
        lab.config(text=_reformat(back.get_local(LABELS[lab])))


def _update_row():
    """Update current row number.

    Returns:
        None
    """
    global CURRENT_ROW
    CURRENT_ROW = CURRENT_ROW + 1


def _update_button():
    """Update button activability depending on the validity of the paths required by the pipeline.

    Returns:
        None
    """
    global BUTTON
    if BUTTON is not None:
        result = back.arevalid_loadbearing_paths()
        if result is True:
            BUTTON.state(['!disabled'])
        else:
            BUTTON.state(['disabled'])


def _reformat(string: list[str] | tuple[str] | str | None) -> str:
    """Reformat given object into a comprehensible string.

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

    selected = Label(root, text=_reformat(back.get_local("output_path")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "output_path"})

    def select_button_click():
        back.set_output_directory()
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

    selected = Label(root, text=_reformat(back.get_local("raw_mot")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_mot"})

    def select_button_click():
        back.set_raw_mots()
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

    selected = Label(root, text=_reformat(back.get_local("raw_trc")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_trc"})

    def select_button_click():
        back.set_raw_trcs()
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


def _setup_row(root: Tk, label_name: str, name_in_json: str, select_action, unselect_action):
    """Set up a line on given window to manage a specific path. Has issues, not to use at the moment.

    Args:
        root: root Tk window.
        label_name: explanatory text of what is aked to the user.
        name_in_json: name of the data to access in the path save.
        select_action: method to call first when select button is clicked. Method should have no parameter.
        unselect_action: method to call first when unselect button is clicked. Method should have no parameter.

    Returns:
        Explanatory Tkinter Label of the path asked.
        Tkinter Label containing current content of the local save.
        Selection Tkinter Button.
        Deselection Tkinter Button.
    """
    raise NotImplemented
    label = Label(root, text=label_name)
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=_reformat(back.get_local(name_in_json)), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: name_in_json})

    def select_button_click():
        select_action()
        _update_labels()
        _update_button()

    select_button = Button(root, text="Select",
                           command=lambda: {select_button_click()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    def unselect_button_click():
        unselect_action()
        _update_labels()
        _update_button()

    unselect_button = Button(root, text="Unselect",
                             command=lambda: {unselect_button_click()})
    unselect_button.grid(row=CURRENT_ROW, column=3, sticky=NW, pady=10)

    _update_row()
    return label, selected, select_button, unselect_button


def _tk_window() -> Tk:
    root = Tk()
    root.title("Path management")
    tbox.set_up_window(root, window_width=800)
    root.columnconfigure(4)

    _setup_output_directory(root)
    _setup_raw_mot(root)
    _setup_raw_trc(root)

    _update_row()
    save_button = Button(root, text="Save to file", default='active',
                         command=lambda: {back.save_to_json()})
    save_button.grid(row=CURRENT_ROW, column=0)

    proceed_button = Button(root, text="Proceed", default='active',
                            command=lambda: {root.destroy()})
    proceed_button.grid(row=CURRENT_ROW, column=1)

    global BUTTON
    BUTTON = proceed_button
    _update_button()

    return root


def missing_loadbearing_path(reason: str) -> None:
    tbox.infobox(reason)
    main()


def main() -> None:
    root = _tk_window()
    root.protocol("WM_DELETE_WINDOW", lambda: tbox.on_closing(root, "Unsaved change will be lost."))
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()


if __name__ == "__main__":
    main()
