from tkinter import *
from tkinter.ttk import *
import resources.tkinter_toolbox as tbox
import resources.paths.paths_back as back

LABELS: dict[Label, str] = {}
CURRENT_ROW = 0


def _update_labels():
    for lab in LABELS:
        lab.config(text=_reformat(back.get_local(LABELS[lab])))


def _update_row():
    global CURRENT_ROW
    CURRENT_ROW = CURRENT_ROW + 1


def _reformat(string_list: list[str] | tuple[str] | str | None) -> str:
    if string_list is None:
        return "empty"
    if isinstance(string_list, str):
        return string_list
    length = len(string_list)
    string = ""
    for i in range(length):
        string = string + string_list[i] + "\n" if i != length - 1 else string + string_list[i]
    return string


def _setup_output_directory(root: Tk) -> (Label, Label, Button, Button):
    label = Label(root, text="Output directory:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=_reformat(back.get_local("output_path")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "output_path"})

    select_button = Button(root, text="Select",
                           command=lambda: {back.set_output_directory(), _update_labels()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    unselect_button = Button(root, text="Unselect",
                             command=lambda: {back.delete_output_directory(), _update_labels()})
    unselect_button.grid(row=CURRENT_ROW, column=3, sticky=NW, pady=10)

    _update_row()
    return label, selected, select_button, unselect_button


def _setup_raw_mot(root: Tk) -> (Label, Label, Button, Button):
    label = Label(root, text="Raw MOT files to process:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=_reformat(back.get_local("raw_mot")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_mot"})

    select_button = Button(root, text="Select",
                           command=lambda: {back.set_raw_mots(), _update_labels()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    unselect_button = Button(root, text="Unselect",
                             command=lambda: {back.delete_raw_mot(), _update_labels()})
    unselect_button.grid(row=CURRENT_ROW, column=3, sticky=NW, pady=10)

    _update_row()
    return label, selected, select_button, unselect_button


def _setup_raw_trc(root: Tk) -> (Label, Label, Button, Button):
    label = Label(root, text="Raw TRC files to process:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=_reformat(back.get_local("raw_trc")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_trc"})

    select_button = Button(root, text="Select",
                           command=lambda: {back.set_raw_trcs(), _update_labels()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    unselect_button = Button(root, text="Unselect",
                             command=lambda: {back.delete_raw_trc(), _update_labels()})
    unselect_button.grid(row=CURRENT_ROW, column=3, sticky=NW, pady=10)

    _update_row()
    return label, selected, select_button, unselect_button


def _setup_row(root: Tk, label_name: str, name_in_json: str, button_txt: str, select_action, unselect_action):
    """Do not use at the moment, not working."""
    raise NotImplemented

    label = Label(root, text=label_name)
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=_reformat(back.get_local(name_in_json)), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: name_in_json})

    select_button = Button(root, text="Select", command={select_action, _update_labels()})
    select_button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    unselect_button = Button(root, text="Unselect",
                             command={unselect_action, _update_labels})
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
    save_button = Button(root, text="Save/Proceed", default='active',
                         command=lambda: {back.save_to_json(), root.destroy()})
    save_button.grid(row=CURRENT_ROW)

    return root


def missing_loadbearing_path(reason: str) -> None:
    tbox.infobox(reason)
    main()


def main() -> None:
    root = _tk_window()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()


if __name__ == "__main__":
    main()
