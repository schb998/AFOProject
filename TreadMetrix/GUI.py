import threading
from tkinter import *
from typing import Literal
import resources.tkinter_toolbox as tbox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from TreadMetrix.full_pipeline import trials_selection
from resources.trial_class import Trial
from data_postprocessing import process as post_processing
from ik_computing import process as compute_ik
from id_computing import process as compute_id
from joint_power_computing import process as compute_jp
from tkinter.constants import DISABLED, NORMAL

# todo: fix plot size
# todo: threading - update plots when data is computed - event handling ??

BUTTONS = {}
PLOTS = {}
TRIALS = {}
CURRENT_TRIAL: str
CANVAS: FigureCanvasTkAgg


def _default_plot():
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.text(0.5, 0.5, f"Nothing to show at the moment", ha='center', va='center', alpha=0.5)
    return fig


DEFAULT_PLOT = _default_plot()


def _blank_plot():
    return Figure(figsize=(5, 5), dpi=100)


BLANK_PLOT = _blank_plot()


def manually_update_buttons(grf: bool = None, ik: bool = None, inverse_d: bool = None, jp: bool = None):
    global BUTTONS
    if grf is not None:
        BUTTONS["grf"]["state"] = NORMAL if grf else DISABLED
    if ik is not None:
        BUTTONS["ik"]["state"] = NORMAL if ik else DISABLED
    if inverse_d is not None:
        BUTTONS["id"]["state"] = NORMAL if inverse_d else DISABLED
    if jp is not None:
        BUTTONS["jp"]["state"] = NORMAL if jp else DISABLED


def update_buttons_for_step(step: Literal["jp", "id", "ik", "grf"] | None = None):
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


def switch_current_trial(new_trial: StringVar):
    global CURRENT_TRIAL
    CURRENT_TRIAL = new_trial
    trial = TRIALS[CURRENT_TRIAL]
    if len(trial.gait_cycles["Right"]) == 0 and len(trial.gait_cycles["Left"]) == 0:
        update_buttons_for_step()
        show_plot(DEFAULT_PLOT)
    else:
        last_gc = trial.gait_cycles["Left"][-1]
        step = "jp" if last_gc.jp is not None \
            else "id" if last_gc.id is not None \
            else "ik" if last_gc.ik is not None \
            else "grf" if last_gc.grf is not None \
            else None
        update_buttons_for_step(step)
        show_plot(PLOTS[CURRENT_TRIAL]["grf"] if step is not None else DEFAULT_PLOT)



def update_grf():
    PLOTS[CURRENT_TRIAL]["grf"] = plot_GRF()
    BUTTONS["grf"]["state"] = NORMAL


def update_ik():
    PLOTS[CURRENT_TRIAL]["ik"] = plot_IK()
    BUTTONS["ik"]["state"] = NORMAL


def update_id():
    PLOTS[CURRENT_TRIAL]["id"] = plot_ID()
    BUTTONS["id"]["state"] = NORMAL


def update_jp():
    PLOTS[CURRENT_TRIAL]["jp"] = plot_JP()
    BUTTONS["jp"]["state"] = NORMAL


def plot_GRF(trial: Trial = None):
    if trial is None:
        trial = TRIALS[CURRENT_TRIAL]
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


def plot_IK(trial: Trial = None):
    if trial is None:
        trial = TRIALS[CURRENT_TRIAL]
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


def plot_ID(trial: Trial = None):
    if trial is None:
        trial = TRIALS[CURRENT_TRIAL]
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


def plot_JP(trial: Trial = None):
    if trial is None:
        trial = TRIALS[CURRENT_TRIAL]
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


def show_plot(fig: Figure):
    CANVAS.figure = fig
    CANVAS.draw()


def gui(output, osim_scaled_model):
    global BUTTONS, PLOTS, TRIALS, CURRENT_TRIAL, CANVAS

    window = Tk()
    window.title('AFO')
    window.geometry("500x800")
    canvas = FigureCanvasTkAgg(DEFAULT_PLOT, master=window)
    canvas.draw()
    CANVAS = canvas

    trial_list = list(TRIALS.keys())
    CURRENT_TRIAL = trial_list[0]
    value_inside = StringVar(window)
    value_inside.set(CURRENT_TRIAL)

    for trial_name in trial_list:
        PLOTS[trial_name] = {"grf": DEFAULT_PLOT, "ik": DEFAULT_PLOT, "id": DEFAULT_PLOT, "jp": DEFAULT_PLOT}
    question_menu = OptionMenu(window, value_inside, *trial_list,
                               command = switch_current_trial)

    # button that displays the plot
    grf_button = Button(master=window,
                        command=lambda: {show_plot(PLOTS[CURRENT_TRIAL]["grf"])},
                        height=2, width=10, text="GRF")
    ik_button = Button(master=window,
                       command=lambda: {show_plot(PLOTS[CURRENT_TRIAL]["ik"])},
                       height=2, width=10, text="IK")
    id_button = Button(master=window,
                       command=lambda: {show_plot(PLOTS[CURRENT_TRIAL]["id"])},
                       height=2, width=10, text="ID")
    jp_button = Button(master=window,
                       command=lambda: {show_plot(PLOTS[CURRENT_TRIAL]["jp"])},
                       height=2, width=10, text="JP")
    BUTTONS = {"grf": grf_button, "ik": ik_button, "id": id_button, "jp": jp_button}
    manually_update_buttons(grf=False, ik=False, inverse_d=False, jp=False)

    current_trial_button = Button(master=window, text="Process",
                                  command=lambda:{pipeline(output, osim_scaled_model)})

    row = 0
    question_menu.grid(column=0, row=row, columnspan=2, sticky="EW")
    current_trial_button.grid(column=2, row=row)
    row += 1
    grf_button.grid(column=0, row=row)
    ik_button.grid(column=1, row=row)
    id_button.grid(column=2, row=row)
    jp_button.grid(column=3, row=row)
    row += 1
    canvas.get_tk_widget().grid(column=0, row=row, columnspan=4, sticky="EW")

    # run the gui
    window.mainloop()


def pipeline(output, osim_scaled_model):
    print(CURRENT_TRIAL)
    trial_to_process = TRIALS[CURRENT_TRIAL]
    if PLOTS[CURRENT_TRIAL]["grf"] == DEFAULT_PLOT:
        post_processing(trial_to_process, save_plot_path=output,
                        save_segmented_path=None,
                        show=False, save_optionals=False)
        update_grf()
        compute_ik(trial_to_process, osim_scaled_model, output, save=False)
        update_ik()
        compute_id(trial_to_process, output, output, osim_scaled_model)
        update_id()
        compute_jp(trial_to_process, output)
        update_jp()
    elif PLOTS[CURRENT_TRIAL]["ik"] == DEFAULT_PLOT:
        compute_ik(trial_to_process, osim_scaled_model, output, save=False)
        update_ik()
        compute_id(trial_to_process, output, output, osim_scaled_model)
        update_id()
        compute_jp(trial_to_process, output)
        update_jp()
    elif PLOTS[CURRENT_TRIAL]["id"] == DEFAULT_PLOT:
        compute_id(trial_to_process, output, output, osim_scaled_model)
        update_id()
        compute_jp(trial_to_process, output)
        update_jp()
    elif PLOTS[CURRENT_TRIAL]["id"] == DEFAULT_PLOT:
        compute_jp(trial_to_process, output)
        update_jp()


def main(output, osim_scaled_model):
    global TRIALS
    TRIALS, _, _ = trials_selection()
    gui(output, osim_scaled_model)

