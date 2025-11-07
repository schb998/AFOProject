from tkinter import *
from tkinter.ttk import *
import TreadMetrix.wip_pipeline.tkinter_toolbox as tbox
import TreadMetrix.wip_pipeline.osim_gestion as osim
import TreadMetrix.paths_back as back

LABELS: dict[Label, str] = {}
CURRENT_ROW = 0


def _update_labels():
    local = back.get_local()
    for lab in LABELS:
        try:
            lab.config(text=_reformat(local[LABELS[lab]]))
        except KeyError:
            lab.config(text=_reformat(None))


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


def _setup_output_directory(root: Tk) -> (Label, Label, Button):
    label = Label(root, text="Output directory:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=_reformat(back.get_local("base_path")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "base_path"})

    button = Button(root, text="Select output directory", command=lambda: {back.set_base_directory(), _update_labels()})
    button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    _update_row()
    return label, selected, button


def _setup_raw_mot(root: Tk) -> (Label, Label, Button):
    label = Label(root, text="Raw MOT files to process:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=_reformat(back.get_local("raw_mot")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_mot"})

    button = Button(root, text="Select raw MOT files to process", command=lambda: {back.set_raw_mots(), _update_labels()})
    button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    _update_row()
    return label, selected, button


def _setup_raw_trc(root: Tk) -> (Label, Label, Button):
    label = Label(root, text="Raw TRC files to process:")
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=_reformat(back.get_local("raw_trc")), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: "raw_trc"})

    button = Button(root, text="Select raw TRC files to process", command=lambda: {back.set_raw_trcs(), _update_labels()})
    button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    _update_row()
    return label, selected, button


def _setup_row(root: Tk, label_name: str, name_in_json: str, button_txt: str, button_action):
    """Do not use at the moment, not working."""
    raise NotImplemented

    label = Label(root, text=label_name)
    label.grid(row=CURRENT_ROW, column=0, sticky=NW, pady=10)

    selected = Label(root, text=_reformat(back.get_local(name_in_json)), background="darkgrey")
    selected.grid(row=CURRENT_ROW, column=1, sticky=NW, pady=10)
    LABELS.update({selected: name_in_json})

    button = Button(root, text=button_txt, command=lambda: {button_action, _update_labels()})
    button.grid(row=CURRENT_ROW, column=2, sticky=NW, pady=10)

    _update_row()
    return label, selected, button


def _osim_button(root: Tk) -> Button:
    osim_button = Button(root, text="Manage local OpenSim paths", command=lambda: {osim.main()})
    osim_button.grid(row=CURRENT_ROW, column=1, sticky=S)
    _update_row()
    return osim_button


def _tk_window() -> Tk:
    root = Tk()
    root.title("Path management")
    tbox.set_up_window(root, window_width=800)
    root.columnconfigure(3)

    _setup_output_directory(root)
    _setup_raw_mot(root)
    _setup_raw_trc(root)

    _osim_button(root)

    save_button = Button(root, text="Save", default='active',
                         command=lambda: {back.save_to_json(), root.destroy()})
    save_button.grid(row=CURRENT_ROW, column=2, sticky=S)

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
