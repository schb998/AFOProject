from tkinter import *
from typing import Literal
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from TreadMetrix.full_pipeline import trials_selection, identify_new_trials_from_dict
from resources.trial_class import Trial
from data_postprocessing import process as post_processing
from ik_computing import process as compute_ik
from id_computing import process as compute_id
from joint_power_computing import process as compute_jp
from tkinter.constants import DISABLED, NORMAL
import resources.paths.paths_access as local
import osim_gestion as osim

# todo: fix plot size
# todo: threading - update plots when data is computed - event handling ??
# todo: fix minor issue with string / stringvar handling of current trial variable
# todo: fix crashing issue when pipeline starts running with no trial to process

BUTTONS = {}
PLOTS = {}
TRIALS = {}
CURRENT_TRIAL: StringVar
TRIAL_DIRECTORY: str
TRIALS_CHOICE: OptionMenu
CANVAS: FigureCanvasTkAgg


def _default_plot() -> Figure:
    """Computes a default plot to show when teh data has not yet been processed"""
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.text(0.5, 0.5, f"Nothing to show at the moment", ha='center', va='center', alpha=0.5)
    return fig


DEFAULT_PLOT = _default_plot()


def _blank_plot() -> Figure:
    """Computes a blank plot to show"""
    return Figure(figsize=(5, 5), dpi=100)


BLANK_PLOT = _blank_plot()


def _manually_update_buttons(grf: bool = None, ik: bool = None, inverse_d: bool = None, jp: bool = None) -> None:
    """Manually activates/deactivates the buttons.

    Args:
        grf: bool, whether the grf button is active
        ik: bool, whether the ik button is active
        inverse_d: bool, whether the id button is active
        jp: bool, whether the jp button is active

    Returns: None

    """
    global BUTTONS
    if grf is not None:
        BUTTONS["grf"]["state"] = NORMAL if grf else DISABLED
    if ik is not None:
        BUTTONS["ik"]["state"] = NORMAL if ik else DISABLED
    if inverse_d is not None:
        BUTTONS["id"]["state"] = NORMAL if inverse_d else DISABLED
    if jp is not None:
        BUTTONS["jp"]["state"] = NORMAL if jp else DISABLED


def _update_buttons_for_step(step: Literal["jp", "id", "ik", "grf"] | None = None) -> None:
    """Updates the button according to the given step of the pipeline

    Args:
        step: str | None, step of the pipeline. Either "jp", "id", "ik", "grf" or None.

    Returns:
        None

    """
    if step is None:
        BUTTONS["grf"]["state"] = DISABLED
        BUTTONS["ik"]["state"] = DISABLED
        BUTTONS["id"]["state"] = DISABLED
        BUTTONS["jp"]["state"] = DISABLED
    match step:
        case "jp":
            BUTTONS["grf"]["state"] = NORMAL
            BUTTONS["ik"]["state"] = NORMAL
            BUTTONS["id"]["state"] = NORMAL
            BUTTONS["jp"]["state"] = NORMAL
        case "id":
            BUTTONS["grf"]["state"] = NORMAL
            BUTTONS["ik"]["state"] = NORMAL
            BUTTONS["id"]["state"] = NORMAL
            BUTTONS["jp"]["state"] = DISABLED
        case "ik":
            BUTTONS["grf"]["state"] = NORMAL
            BUTTONS["ik"]["state"] = NORMAL
            BUTTONS["id"]["state"] = DISABLED
            BUTTONS["jp"]["state"] = DISABLED
        case "grf":
            BUTTONS["grf"]["state"] = NORMAL
            BUTTONS["ik"]["state"] = DISABLED
            BUTTONS["id"]["state"] = DISABLED
            BUTTONS["jp"]["state"] = DISABLED
            BUTTONS["grf"]["state"]  = NORMAL
        case _:
            BUTTONS["grf"]["state"] = DISABLED
            BUTTONS["ik"]["state"] = DISABLED
            BUTTONS["id"]["state"] = DISABLED
            BUTTONS["jp"]["state"] = DISABLED


def _switch_current_trial_from_stringvar(new_trial: StringVar) -> None:
    """Switch the current trial from a StingVar value

    Args:
        new_trial: StingVar, value of the current trial

    Returns: None

    """
    _switch_current_trial(new_trial.get())


def _switch_current_trial(new_trial: str):
    global CURRENT_TRIAL
    CURRENT_TRIAL.set(new_trial)
    trial = TRIALS[new_trial]
    if len(trial.gait_cycles["Right"]) == 0 and len(trial.gait_cycles["Left"]) == 0:
        _update_buttons_for_step()
        _show_plot(DEFAULT_PLOT)
    else:
        last_gc = trial.gait_cycles["Left"][-1]
        step = "jp" if last_gc.jp is not None \
            else "id" if last_gc.id is not None \
            else "ik" if last_gc.ik is not None \
            else "grf" if last_gc.grf is not None \
            else None
        _update_buttons_for_step(step)
        _show_plot(PLOTS[new_trial]["grf"] if step is not None else DEFAULT_PLOT)


def _update_grf() -> None:
    """Update the grf plot of the current trial"""
    PLOTS[CURRENT_TRIAL.get()]["grf"] = _plot_GRF()
    BUTTONS["grf"]["state"] = NORMAL


def _update_ik() -> None:
    """Update the ik plot of the current trial"""
    PLOTS[CURRENT_TRIAL.get()]["ik"] = _plot_IK()
    BUTTONS["ik"]["state"] = NORMAL


def _update_id() -> None:
    """Update the id plot of the current trial"""
    PLOTS[CURRENT_TRIAL.get()]["id"] = _plot_ID()
    BUTTONS["id"]["state"] = NORMAL


def _update_jp() -> None:
    """Update the jp plot of the current trial"""
    PLOTS[CURRENT_TRIAL.get()]["jp"] = _plot_JP()
    BUTTONS["jp"]["state"] = NORMAL


def _plot_GRF(trial: Trial = None) -> Figure:
    """Makes a plot of the trial's GRF"""
    if trial is None:
        trial = TRIALS[CURRENT_TRIAL.get()]
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.set_ylabel("GRF (N)")
    plot1.set_xlabel("% of the gait cycle")

    r_column = 'ground_force2_vy'
    l_column = 'ground_force1_vy'

    try:
        for cycle in trial.gait_cycles["Right"]:
            grf = cycle.grf.data
            start_time = grf['time'].iloc[0]
            end_time = grf['time'].iloc[-1]
            time_scale = [100 * (i - start_time) / (end_time - start_time) for i in grf['time'].values]
            plot1.plot(time_scale, grf[r_column], label="Ground Force (R)", color='orange', alpha=0.3)

        for cycle in trial.gait_cycles["Left"]:
            grf = cycle.grf.data
            start_time = grf['time'].iloc[0]
            end_time = grf['time'].iloc[-1]
            time_scale = [100 * (i - start_time) / (end_time - start_time) for i in grf['time'].values]
            plot1.plot(time_scale, grf[l_column], label="Ground Force (L)", color='green', alpha=0.3)

    except AttributeError:
        plot1.text(0.5, 0.5, f"No GRF data yet", ha='center', va='center', alpha=0.5)
        return fig

    except Exception as e:
        plot1.clear()
        plot1.text(0.5, 0.5, f"Error: {getattr(e, 'message', repr(e))}", ha='center', va='center')
        return fig

    handles, labels = plot1.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    fig.legend(handle_list, label_list)
    return fig


def _plot_IK(trial: Trial = None)-> Figure:
    """Makes a plot of the trial's IK"""
    if trial is None:
        trial = TRIALS[CURRENT_TRIAL.get()]
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.set_ylabel("Ankle angle (insert unit)")
    plot1.set_xlabel("% of the gait cycle")

    r_column = 'ankle_angle_r'
    l_column = 'ankle_angle_l'

    try:
        for cycle in trial.gait_cycles["Right"]:
            ik = cycle.ik.data
            start_time = ik['time'].iloc[0]
            end_time = ik['time'].iloc[-1]
            time_scale = [100 * (i - start_time) / (end_time - start_time) for i in ik['time'].values]
            plot1.plot(time_scale, ik[r_column], label="Ankle angle (R)", color='orange', alpha=0.3)

        for cycle in trial.gait_cycles["Left"]:
            ik = cycle.ik.data
            start_time = ik['time'].iloc[0]
            end_time = ik['time'].iloc[-1]
            time_scale = [100 * (i - start_time) / (end_time - start_time) for i in ik['time'].values]
            plot1.plot(time_scale, ik[l_column], label="Ankle angle (L)", color='green', alpha=0.3)

    except AttributeError:
        plot1.text(0.5, 0.5, f"No IK data yet", ha='center', va='center', alpha=0.5)
        return fig

    except Exception as e:
        plot1.clear()
        plot1.text(0.5, 0.5, f"Error: {getattr(e, 'message', repr(e))}", ha='center', va='center')
        return fig

    handles, labels = plot1.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    fig.legend(handle_list, label_list)
    return fig


def _plot_ID(trial: Trial = None)-> Figure:
    """Makes a plot of the trial's ID"""
    if trial is None:
        trial = TRIALS[CURRENT_TRIAL.get()]
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.set_ylabel("Ankle angle moment (insert unit)")
    plot1.set_xlabel("% of the gait cycle")

    r_column = 'ankle_angle_r_moment'
    l_column = 'ankle_angle_l_moment'

    try:
        for cycle in trial.gait_cycles["Right"]:
            gait_id = cycle.id.data
            start_time = gait_id['time'].iloc[0]
            end_time = gait_id['time'].iloc[-1]
            time_scale = [100 * (i - start_time) / (end_time - start_time) for i in gait_id['time'].values]
            plot1.plot(time_scale, gait_id[r_column], label="Ankle angle moment (R)", color='orange', alpha=0.3)

        for cycle in trial.gait_cycles["Left"]:
            gait_id = cycle.id.data
            start_time = gait_id['time'].iloc[0]
            end_time = gait_id['time'].iloc[-1]
            time_scale = [100 * (i - start_time) / (end_time - start_time) for i in gait_id['time'].values]
            plot1.plot(time_scale, gait_id[l_column], label="Ankle angle moment (L)", color='green', alpha=0.3)

    except AttributeError:
        plot1.text(0.5, 0.5, f"No ID data yet", ha='center', va='center', alpha=0.5)
        return fig

    except Exception as e:
        plot1.clear()
        plot1.text(0.5, 0.5, f"Error: {getattr(e, 'message', repr(e))}", ha='center', va='center')
        return fig

    handles, labels = plot1.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)

    fig.legend(handle_list, label_list)
    return fig


def _plot_JP(trial: Trial = None)-> Figure:
    """Makes a plot of the trial's JP"""
    if trial is None:
        trial = TRIALS[CURRENT_TRIAL.get()]
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.set_ylabel("Ankle angle power (insert unit)")
    plot1.set_xlabel("% of the gait cycle")

    r_column = 'ankle_angle_r_power'
    l_column = 'ankle_angle_l_power'

    try:
        for cycle in trial.gait_cycles["Right"]:
            jp = cycle.jp.data
            start_time = jp['time'].iloc[0]
            end_time = jp['time'].iloc[-1]
            time_scale = [100 * (i - start_time) / (end_time - start_time) for i in jp['time'].values]
            plot1.plot(time_scale, jp[r_column], label="Ankle angle power (R)", color='orange', alpha=0.3)

        for cycle in trial.gait_cycles["Left"]:
            jp = cycle.jp.data
            start_time = jp['time'].iloc[0]
            end_time = jp['time'].iloc[-1]
            time_scale = [100 * (i - start_time) / (end_time - start_time) for i in jp['time'].values]
            plot1.plot(time_scale, jp[l_column], label="Ankle angle power (L)", color='green', alpha=0.3)

    except AttributeError:
        plot1.text(0.5, 0.5, f"Missing JP data", ha='center', va='center', alpha=0.5)
        return fig

    except Exception as e:
        plot1.clear()
        plot1.text(0.5, 0.5, f"Error: {getattr(e, 'message', repr(e))}", ha='center', va='center')
        return fig

    handles, labels = plot1.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)

    fig.legend(handle_list, label_list)
    return fig


def _show_plot(fig: Figure):
    """Sets the given plot into the window."""
    CANVAS.figure = fig
    CANVAS.draw()


def _gui(output, osim_scaled_model) -> None:
    """GUI setup.

    Parameters:
        output: str, output directory of the pipeline
        osim_scaled_model: str, path to the scaled model

    returns: None
    """
    global BUTTONS, PLOTS, TRIALS, CURRENT_TRIAL, CANVAS, TRIALS_CHOICE

    window = Tk()
    window.title('AFO')
    window.geometry("500x800")
    canvas = FigureCanvasTkAgg(DEFAULT_PLOT, master=window)
    canvas.draw()
    CANVAS = canvas

    trial_list = list(TRIALS.keys())

    CURRENT_TRIAL = StringVar(window)
    CURRENT_TRIAL.set(trial_list[0])

    for trial_name in trial_list:
        PLOTS[trial_name] = {"grf": DEFAULT_PLOT, "ik": DEFAULT_PLOT, "id": DEFAULT_PLOT, "jp": DEFAULT_PLOT}

    question_menu = OptionMenu(window, CURRENT_TRIAL, *trial_list,
                               command = _switch_current_trial_from_stringvar)
    TRIALS_CHOICE = question_menu

    # button that displays the plot
    grf_button = Button(master=window,
                        command=lambda: {_show_plot(PLOTS[CURRENT_TRIAL.get()]["grf"])},
                        height=2, width=10, text="GRF")
    ik_button = Button(master=window,
                       command=lambda: {_show_plot(PLOTS[CURRENT_TRIAL.get()]["ik"])},
                       height=2, width=10, text="IK")
    id_button = Button(master=window,
                       command=lambda: {_show_plot(PLOTS[CURRENT_TRIAL.get()]["id"])},
                       height=2, width=10, text="ID")
    jp_button = Button(master=window,
                       command=lambda: {_show_plot(PLOTS[CURRENT_TRIAL.get()]["jp"])},
                       height=2, width=10, text="JP")
    BUTTONS = {"grf": grf_button, "ik": ik_button, "id": id_button, "jp": jp_button}
    _manually_update_buttons(grf=False, ik=False, inverse_d=False, jp=False)

    current_trial_button = Button(master=window, text="Process",
                                  command=lambda:{_pipeline(output, osim_scaled_model)})

    row = 0
    question_menu.grid(column=0, row=row, columnspan=2, sticky="EW")
    current_trial_button.grid(column=2, row=row)
    if TRIAL_DIRECTORY is not None:
        search_trials_button = Button(master=window, text="Search new trials", command=lambda:{_update_trials()})
        search_trials_button.grid(column=3, row=row)
    row += 1
    grf_button.grid(column=0, row=row)
    ik_button.grid(column=1, row=row)
    id_button.grid(column=2, row=row)
    jp_button.grid(column=3, row=row)
    row += 1
    canvas.get_tk_widget().grid(column=0, row=row, columnspan=4, sticky="EW")

    # run the gui
    window.mainloop()


def _update_trials() -> None:
    """Check if new trials are to be processed and add them to the trials list"""
    new_trials = identify_new_trials_from_dict(TRIAL_DIRECTORY, list(TRIALS.keys()))
    new_trials_names = list(new_trials.keys())
    TRIALS.update(new_trials)
    menu = TRIALS_CHOICE.children["menu"]
    for trial_name in new_trials_names:
        PLOTS[trial_name] = {"grf": DEFAULT_PLOT, "ik": DEFAULT_PLOT, "id": DEFAULT_PLOT, "jp": DEFAULT_PLOT}
        menu.add_command(label=trial_name, command=lambda: {_switch_current_trial(trial_name)})


def _pipeline(output, osim_scaled_model) -> None:
    """Runs the pipeline for the current trial, using the output directory and the given scaled model"""
    current_trial = CURRENT_TRIAL.get()
    print(current_trial)
    trial_to_process = TRIALS[current_trial]
    if PLOTS[current_trial]["grf"] == DEFAULT_PLOT:
        post_processing(trial_to_process, save_plot_path=output,
                        save_segmented_path=None,
                        show=False, save_optionals=False)
        _update_grf()
        compute_ik(trial_to_process, osim_scaled_model, output)
        _update_ik()
        compute_id(trial_to_process, output, output, osim_scaled_model)
        _update_id()
        compute_jp(trial_to_process, output)
        _update_jp()
    elif PLOTS[current_trial]["ik"] == DEFAULT_PLOT:
        compute_ik(trial_to_process, osim_scaled_model, output)
        _update_ik()
        compute_id(trial_to_process, output, output, osim_scaled_model)
        _update_id()
        compute_jp(trial_to_process, output)
        _update_jp()
    elif PLOTS[current_trial]["id"] == DEFAULT_PLOT:
        compute_id(trial_to_process, output, output, osim_scaled_model)
        _update_id()
        compute_jp(trial_to_process, output)
        _update_jp()
    elif PLOTS[current_trial]["jp"] == DEFAULT_PLOT:
        compute_jp(trial_to_process, output)
        _update_jp()
    if TRIAL_DIRECTORY is not None:
        _update_trials()


def main() -> None:
    """Main loop of the GUI pipeline."""
    global TRIALS, TRIAL_DIRECTORY
    if not local.call_quick_setup():
        local.main_gui()
        osim.main()
    TRIALS, TRIAL_DIRECTORY = trials_selection()
    _gui(local.get_output_path(), local.get_scaled_model_file())


if __name__ == "__main__":
    main()
