"""
interactive_gait_selector.py
=============================
Interactive Gait Event & Cycle Selector GUI for the TreadMetrix pipeline.

Allows visual inspection of vertical GRF signals and TRC heel marker
trajectories, interactive addition/deletion/adjustment of heel strike
and toe-off timestamps, manual entry/editing of trial speed and slope,
and seamless exporting/segmentation of gait cycles into OpenSim TRC and MOT files.
"""

import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from scipy.signal import butter, filtfilt

# Path resolution for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def get_grf_columns(df):
    """Find vertical force column names for Right (Plate 5/2) and Left (Plate 4/1)."""
    cols = df.columns.tolist()
    r_col = None
    l_col = None
    for c in ['ground_force5_vy', 'ground_force2_vy', 'force5_vy', 'force2_vy']:
        if c in cols:
            r_col = c
            break
    for c in ['ground_force4_vy', 'ground_force1_vy', 'force4_vy', 'force1_vy']:
        if c in cols:
            l_col = c
            break
    return r_col, l_col


def detect_toe_offs_from_signal(fy_signal, t_array, hs_times, threshold=15.0):
    """Fallback detector: find first frame after each HS where force drops below threshold."""
    to_times = []
    if len(t_array) < 2 or len(hs_times) == 0:
        return to_times
    dt = float(np.mean(np.diff(t_array)))
    fs = 1.0 / dt
    for hs_t in hs_times:
        idx_hs = int(np.argmin(np.abs(t_array - hs_t)))
        search_max = min(len(fy_signal) - 1, idx_hs + int(1.2 * fs))
        sub = fy_signal[idx_hs:search_max]
        below = np.where(sub < threshold)[0]
        if len(below) > 0:
            idx_to = idx_hs + below[0]
            to_times.append(float(t_array[idx_to]))
    return sorted(list(set(to_times)))


class GaitEventSelectorGUI:
    def __init__(self, root, trial=None, initial_r_hs=None, initial_l_hs=None,
                 initial_r_to=None, initial_l_to=None,
                 speed=0.0, slope=0.0, postproc_version='v2'):
        self.root = root
        self.root.title("Interactive Gait Event & Cycle Selector (TreadMetrix)")
        self.root.geometry("1560x960")

        self.trial = trial
        self.postproc_version = postproc_version
        
        # Result parameters
        self.result_speed = float(speed)
        self.result_slope = float(slope)
        self.finished = False

        # Active state flags
        self.active_side = "Right"       # "Right" or "Left"
        self.active_event_type = "HS"     # "HS" (Heel Strike) or "TO" (Toe Off)
        self.active_mode = "Click"        # "Click" or "Drag"

        # Data storage
        self.mot_grf = None
        self.trc_obj = None
        self.t_grf = None
        self.r_fy = None
        self.l_fy = None
        self.r_fy_filt = None
        self.l_fy_filt = None
        self.t_trc = None
        self.r_heel_y = None
        self.l_heel_y = None

        # Event lists (timestamps in seconds)
        self.r_hs_times = list(initial_r_hs) if initial_r_hs is not None else []
        self.l_hs_times = list(initial_l_hs) if initial_l_hs is not None else []
        self.r_to_times = list(initial_r_to) if initial_r_to is not None else []
        self.l_to_times = list(initial_l_to) if initial_l_to is not None else []

        # View Window
        self.window_size = 10.0
        self.current_t_start = 0.0

        # UI Variables for Speed & Slope
        self.speed_var = tk.StringVar(master=self.root, value=str(speed))
        self.slope_var = tk.StringVar(master=self.root, value=str(slope))

        # Build UI
        self._build_controls()
        self._build_main_layout()

        # Load trial data
        if self.trial is not None:
            self._load_trial_data()

    def _build_controls(self):
        top_frame = tk.Frame(self.root, bg="#f0f0f0", pady=6, padx=8)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top_frame, text="Gait Cycle Selector", font=("Helvetica", 14, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=6)

        # --- Active Foot Buttons ---
        tk.Label(top_frame, text="Foot:", font=("Helvetica", 10, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=(10, 2))
        
        self.btn_right = tk.Button(top_frame, text="🔴 RIGHT FOOT", font=("Helvetica", 9, "bold"),
                                   bg="#ffcccc", fg="darkred", activebackground="#ff9999",
                                   relief=tk.SUNKEN, bd=3, command=lambda: self._set_active_foot("Right"))
        self.btn_right.pack(side=tk.LEFT, padx=2)

        self.btn_left = tk.Button(top_frame, text="🔵 LEFT FOOT", font=("Helvetica", 9, "bold"),
                                  bg="#e6f2ff", fg="darkblue", activebackground="#cce6ff",
                                  relief=tk.RAISED, bd=1, command=lambda: self._set_active_foot("Left"))
        self.btn_left.pack(side=tk.LEFT, padx=2)

        # --- Active Event Type Buttons ---
        tk.Label(top_frame, text="Event:", font=("Helvetica", 10, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=(10, 2))
        
        self.btn_hs = tk.Button(top_frame, text="👠 Heel Strike (HS)", font=("Helvetica", 9, "bold"),
                                bg="#e0e0e0", relief=tk.SUNKEN, bd=3, command=lambda: self._set_event_type("HS"))
        self.btn_hs.pack(side=tk.LEFT, padx=2)

        self.btn_to = tk.Button(top_frame, text="🦶 Toe Off (TO)", font=("Helvetica", 9, "bold"),
                                bg="#f0f0f0", relief=tk.RAISED, bd=1, command=lambda: self._set_event_type("TO"))
        self.btn_to.pack(side=tk.LEFT, padx=2)

        # --- Mode Buttons ---
        tk.Label(top_frame, text="Mode:", font=("Helvetica", 10, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=(10, 2))
        
        self.btn_mode_click = tk.Button(top_frame, text="Click Points", font=("Helvetica", 9),
                                         relief=tk.SUNKEN, bd=2, command=lambda: self._set_mode("Click"))
        self.btn_mode_click.pack(side=tk.LEFT, padx=2)

        self.btn_mode_drag = tk.Button(top_frame, text="Drag Region", font=("Helvetica", 9),
                                        relief=tk.RAISED, bd=1, command=lambda: self._set_mode("Drag"))
        self.btn_mode_drag.pack(side=tk.LEFT, padx=2)

        # --- Speed & Slope Inputs ---
        tk.Label(top_frame, text="Speed (mph):", font=("Helvetica", 9, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=(10, 2))
        speed_entry = ttk.Entry(top_frame, textvariable=self.speed_var, width=5)
        speed_entry.pack(side=tk.LEFT, padx=2)

        tk.Label(top_frame, text="Slope (%):", font=("Helvetica", 9, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=(6, 2))
        slope_entry = ttk.Entry(top_frame, textvariable=self.slope_var, width=5)
        slope_entry.pack(side=tk.LEFT, padx=2)

        # --- Clear / Reset Action Buttons ---
        btn_clear_side = ttk.Button(top_frame, text="Clear Active Foot", command=self._clear_active_foot)
        btn_clear_side.pack(side=tk.LEFT, padx=6)

        btn_auto = ttk.Button(top_frame, text="Auto Detect All", command=self._auto_detect_events)
        btn_auto.pack(side=tk.LEFT, padx=3)

        # --- Finish & Segment Button ---
        btn_finish = ttk.Button(top_frame, text="🚀 Finish & Segment", command=self._on_finish)
        btn_finish.pack(side=tk.RIGHT, padx=10)

        # --- Help / Status Label ---
        self.lbl_help = ttk.Label(top_frame, text="Status: Ready", font=("Helvetica", 9, "italic"), foreground="navy")
        self.lbl_help.pack(side=tk.RIGHT, padx=10)

        # --- Time Window Scroll Frame ---
        scroll_frame = ttk.Frame(self.root, padding=4)
        scroll_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(scroll_frame, text="Time Window (s):").pack(side=tk.LEFT, padx=5)
        self.slider = ttk.Scale(scroll_frame, from_=0.0, to_=45.0, value=0.0, orient=tk.HORIZONTAL, command=self._on_slider_move)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.lbl_window = ttk.Label(scroll_frame, text="View Window: 0.0s - 10.0s", font=("Helvetica", 10, "bold"))
        self.lbl_window.pack(side=tk.RIGHT, padx=10)

    def _build_main_layout(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Plot Frame (Left)
        plot_frame = ttk.Frame(paned)
        paned.add(plot_frame, weight=4)

        self.fig, (self.ax_grf, self.ax_trc) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=100)
        self.fig.subplots_adjust(hspace=0.25, left=0.07, right=0.98, top=0.92, bottom=0.08)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Connect click events
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

        # SpanSelector widget for Drag Region mode
        self.span = SpanSelector(
            self.ax_grf, self._on_span_select, 'horizontal', useblit=True,
            props=dict(alpha=0.35, facecolor='red'), interactive=True
        )
        self.span.set_active(False)

        # Side Panel (Right) for Cycles Table
        side_panel = ttk.Frame(paned, padding=8)
        paned.add(side_panel, weight=1)

        ttk.Label(side_panel, text="Detected Gait Cycles", font=("Helvetica", 11, "bold")).pack(side=tk.TOP, anchor="w", pady=5)

        columns = ("side", "stride", "start", "end", "dur")
        self.tree = ttk.Treeview(side_panel, columns=columns, show="headings", height=22)
        self.tree.heading("side", text="Side")
        self.tree.heading("stride", text="#")
        self.tree.heading("start", text="Start(s)")
        self.tree.heading("end", text="End(s)")
        self.tree.heading("dur", text="Dur(s)")

        self.tree.column("side", width=55, anchor="center")
        self.tree.column("stride", width=35, anchor="center")
        self.tree.column("start", width=65, anchor="center")
        self.tree.column("end", width=65, anchor="center")
        self.tree.column("dur", width=60, anchor="center")

        tree_scroll = ttk.Scrollbar(side_panel, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _set_active_foot(self, foot):
        self.active_side = foot
        if foot == "Right":
            self.btn_right.config(relief=tk.SUNKEN, bd=3, bg="#ffcccc")
            self.btn_left.config(relief=tk.RAISED, bd=1, bg="#e6f2ff")
        else:
            self.btn_right.config(relief=tk.RAISED, bd=1, bg="#ffcccc")
            self.btn_left.config(relief=tk.SUNKEN, bd=3, bg="#e6f2ff")
        print(f"Foot switched to: {self.active_side}")
        self._update_plot()

    def _set_event_type(self, etype):
        self.active_event_type = etype
        if etype == "HS":
            self.btn_hs.config(relief=tk.SUNKEN, bd=3, bg="#d0d0d0")
            self.btn_to.config(relief=tk.RAISED, bd=1, bg="#f0f0f0")
        else:
            self.btn_hs.config(relief=tk.RAISED, bd=1, bg="#f0f0f0")
            self.btn_to.config(relief=tk.SUNKEN, bd=3, bg="#d0d0d0")
        print(f"Event type switched to: {self.active_event_type}")
        self._update_plot()

    def _set_mode(self, mode):
        self.active_mode = mode
        if mode == "Drag":
            self.btn_mode_click.config(relief=tk.RAISED, bd=1)
            self.btn_mode_drag.config(relief=tk.SUNKEN, bd=3)
            self.span.set_active(True)
            self.lbl_help.config(text="Mode: Drag Region across graph to select 1 cycle")
        else:
            self.btn_mode_click.config(relief=tk.SUNKEN, bd=3)
            self.btn_mode_drag.config(relief=tk.RAISED, bd=1)
            self.span.set_active(False)
            self.lbl_help.config(text="Mode: Left-Click to Add Point | Right-Click to Delete Point")

    def _load_trial_data(self):
        try:
            self.mot_grf = self.trial.corrected_grf if self.trial.corrected_grf is not None else self.trial.grf
            self.trc_obj = self.trial.trc

            df_grf = self.mot_grf.data
            self.t_grf = df_grf['time'].values
            fs_grf = 1.0 / np.mean(np.diff(self.t_grf))

            # Column mapping for GRF
            r_col, l_col = get_grf_columns(df_grf)
            
            self.r_fy = df_grf[r_col].values if r_col and r_col in df_grf.columns else np.zeros(len(self.t_grf))
            self.l_fy = df_grf[l_col].values if l_col and l_col in df_grf.columns else np.zeros(len(self.t_grf))

            # Low pass filter for GRF visualization
            b_g, a_g = butter(4, min(0.99, 10.0 / (0.5 * fs_grf)), btype='low')
            self.r_fy_filt = filtfilt(b_g, a_g, self.r_fy)
            self.l_fy_filt = filtfilt(b_g, a_g, self.l_fy)

            # TRC Marker data
            if self.trc_obj is not None:
                df_trc = self.trc_obj.data
                self.t_trc = df_trc['Time'].values

                r_heel_col, l_heel_col = None, None
                if hasattr(self.trc_obj, 'marker_dict'):
                    for m_name, cols in self.trc_obj.marker_dict.items():
                        if m_name.upper() in ['RCAL', 'RHEEL', 'HEEL_R']:
                            r_heel_col = cols[1]
                        elif m_name.upper() in ['LCAL', 'LHEEL', 'HEEL_L']:
                            l_heel_col = cols[1]

                self.r_heel_y = df_trc[r_heel_col].values if r_heel_col and r_heel_col in df_trc.columns else np.zeros(len(self.t_trc))
                self.l_heel_y = df_trc[l_heel_col].values if l_heel_col and l_heel_col in df_trc.columns else np.zeros(len(self.t_trc))

                t_max = max(0.0, self.t_trc[-1] - self.window_size)
                self.slider.config(from_=self.t_trc[0], to=t_max)
                self.current_t_start = self.t_trc[0]
            else:
                self.t_trc = self.t_grf
                self.r_heel_y = np.zeros(len(self.t_trc))
                self.l_heel_y = np.zeros(len(self.t_trc))

            # Auto-detect if events are missing
            if len(self.r_hs_times) == 0 or len(self.l_hs_times) == 0 or len(self.r_to_times) == 0 or len(self.l_to_times) == 0:
                self._auto_detect_events()
            else:
                self._update_plot()

        except Exception as e:
            messagebox.showerror("Error Loading Data", str(e))

    def _auto_detect_events(self):
        if self.t_grf is None:
            return
        fs_grf = 1.0 / np.mean(np.diff(self.t_grf))

        if self.postproc_version == 'v2':
            from data_postprocessing_V2 import detect_heel_strikes_V2, detect_toe_offs_V2
            hs_dict = detect_heel_strikes_V2(self.mot_grf)
            to_dict = detect_toe_offs_V2(self.mot_grf)
        else:
            from data_postprocessing import detect_heel_strikes, detect_toe_offs
            hs_dict = detect_heel_strikes(self.mot_grf, fs_grf)
            to_dict = detect_toe_offs(self.mot_grf, fs_grf)

        self.r_hs_times = [float(self.t_grf[idx]) for idx in hs_dict['R'] if idx < len(self.t_grf)]
        self.l_hs_times = [float(self.t_grf[idx]) for idx in hs_dict['L'] if idx < len(self.t_grf)]
        
        r_to = [float(self.t_grf[idx]) for idx in to_dict['R'] if idx < len(self.t_grf)]
        l_to = [float(self.t_grf[idx]) for idx in to_dict['L'] if idx < len(self.t_grf)]

        # Fallback toe-off detection if needed
        if len(r_to) == 0 and len(self.r_hs_times) > 0:
            r_to = detect_toe_offs_from_signal(self.r_fy_filt, self.t_grf, self.r_hs_times)
        if len(l_to) == 0 and len(self.l_hs_times) > 0:
            l_to = detect_toe_offs_from_signal(self.l_fy_filt, self.t_grf, self.l_hs_times)

        self.r_to_times = r_to
        self.l_to_times = l_to

        print(f"Auto-detected events: Right HS={len(self.r_hs_times)}, Right TO={len(self.r_to_times)}, Left HS={len(self.l_hs_times)}, Left TO={len(self.l_to_times)}")
        self._update_plot()

    def _on_slider_move(self, val):
        self.current_t_start = float(val)
        self._update_plot()

    def _get_target_event_list(self):
        if self.active_side == "Right":
            return self.r_hs_times if self.active_event_type == "HS" else self.r_to_times
        else:
            return self.l_hs_times if self.active_event_type == "HS" else self.l_to_times

    def _on_canvas_click(self, event):
        if self.active_mode == "Drag":
            return
        if event.inaxes not in [self.ax_grf, self.ax_trc] or event.xdata is None:
            return

        click_t = float(event.xdata)
        side = self.active_side
        etype = "Heel Strike" if self.active_event_type == "HS" else "Toe Off"

        target_list = self._get_target_event_list()

        # Right-Click -> Delete nearest event (within 0.5s)
        if event.button == 3:
            if len(target_list) > 0:
                diffs = [abs(t - click_t) for t in target_list]
                min_idx = int(np.argmin(diffs))
                if diffs[min_idx] <= 0.5:
                    removed = target_list.pop(min_idx)
                    print(f"Deleted {side} {etype} at t={removed:.3f}s")
                    self.lbl_help.config(text=f"Deleted {side} {etype} at t={removed:.3f}s")
                    self._update_plot()
            return

        # Left-Click -> Add exact timestamp at click location
        target_list.append(click_t)
        target_list.sort()
        print(f"Added {side} {etype} at t={click_t:.3f}s")
        self.lbl_help.config(text=f"Added {side} {etype} at t={click_t:.3f}s")
        self._update_plot()

    def _on_span_select(self, xmin, xmax):
        if self.active_mode != "Drag":
            return
        t1, t2 = min(xmin, xmax), max(xmin, xmax)
        if (t2 - t1) < 0.3:
            return

        target_list = self.r_hs_times if self.active_side == "Right" else self.l_hs_times

        target_list.append(t1)
        target_list.append(t2)
        target_list.sort()
        print(f"Region Selected for {self.active_side}: {t1:.3f}s to {t2:.3f}s")
        self._update_plot()

    def _clear_active_foot(self):
        side = self.active_side
        print(f"Clearing all HS and TO events for {side} foot")
        if side == "Right":
            self.r_hs_times = []
            self.r_to_times = []
        else:
            self.l_hs_times = []
            self.l_to_times = []
        self.lbl_help.config(text=f"Cleared all events for {side} foot")
        self._update_plot()

    def _get_paired_cycles(self):
        t_trc_start = float(self.t_trc[0]) if self.t_trc is not None else 0.0
        t_trc_end = float(self.t_trc[-1]) if self.t_trc is not None else 100.0

        def pair(hs_times):
            cycles = []
            hs_sorted = sorted(list(set(hs_times)))
            for i in range(len(hs_sorted) - 1):
                t1 = float(hs_sorted[i])
                t2 = float(hs_sorted[i+1])
                dur = t2 - t1
                if t1 < t_trc_start or t2 > t_trc_end:
                    continue
                if dur > 0.05:
                    cycles.append((t1, t2, dur))
            return cycles

        return pair(self.r_hs_times), pair(self.l_hs_times)

    def _update_plot(self):
        if self.t_grf is None:
            return

        t_start = self.current_t_start
        t_end = t_start + self.window_size

        r_cycles, l_cycles = self._get_paired_cycles()

        self.lbl_window.config(
            text=f"View Window: {t_start:.1f}s - {t_end:.1f}s | "
                 f"Right Cycles: {len(r_cycles)} | "
                 f"Left Cycles: {len(l_cycles)}"
        )

        # Refresh Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        for idx, (t1, t2, dur) in enumerate(r_cycles):
            self.tree.insert("", "end", values=("Right", idx, f"{t1:.2f}", f"{t2:.2f}", f"{dur:.2f}"))
        for idx, (t1, t2, dur) in enumerate(l_cycles):
            self.tree.insert("", "end", values=("Left", idx, f"{t1:.2f}", f"{t2:.2f}", f"{dur:.2f}"))

        self.ax_grf.clear()
        self.ax_trc.clear()

        mask_g = (self.t_grf >= t_start) & (self.t_grf <= t_end)
        mask_t = (self.t_trc >= t_start) & (self.t_trc <= t_end)

        # Highlight paired cycles
        for idx, (c_start, c_end, dur) in enumerate(r_cycles):
            if c_end >= t_start and c_start <= t_end:
                self.ax_grf.axvspan(c_start, c_end, facecolor='#ff9999', alpha=0.30, edgecolor='red', linestyle='--')
                self.ax_trc.axvspan(c_start, c_end, facecolor='#ff9999', alpha=0.30, edgecolor='red', linestyle='--')
                mid_t = max(t_start + 0.2, (c_start + c_end) / 2.0)
                max_y = max(np.max(self.r_fy_filt[mask_g]) if np.any(mask_g) else 100, 300)
                self.ax_grf.text(mid_t, max_y * 0.8, f"R Cycle {idx}\n({dur:.2f}s)", color='darkred', fontweight='bold', fontsize=9, ha='center')

        for idx, (c_start, c_end, dur) in enumerate(l_cycles):
            if c_end >= t_start and c_start <= t_end:
                self.ax_grf.axvspan(c_start, c_end, facecolor='#99ccff', alpha=0.30, edgecolor='blue', linestyle='--')
                self.ax_trc.axvspan(c_start, c_end, facecolor='#99ccff', alpha=0.30, edgecolor='blue', linestyle='--')
                mid_t = max(t_start + 0.2, (c_start + c_end) / 2.0)
                max_y = max(np.max(self.l_fy_filt[mask_g]) if np.any(mask_g) else 100, 300)
                self.ax_grf.text(mid_t, max_y * 0.9, f"L Cycle {idx}\n({dur:.2f}s)", color='darkblue', fontweight='bold', fontsize=9, ha='center')

        # 1. GRF Plot
        self.ax_grf.plot(self.t_grf[mask_g], self.r_fy_filt[mask_g], color='orange', linewidth=1.8, label='Right Vertical Force')
        self.ax_grf.plot(self.t_grf[mask_g], self.l_fy_filt[mask_g], color='green', linewidth=1.8, label='Left Vertical Force')
        self.ax_grf.set_ylabel("Force (N)")
        
        etype_str = "Heel Strike (HS)" if self.active_event_type == "HS" else "Toe Off (TO)"
        active_info = f"ACTIVE: {self.active_side.upper()} Foot | Event: {etype_str}"
        self.ax_grf.set_title(f"Ground Reaction Force [{active_info}] (Left-Click Add, Right-Click Delete)", fontsize=11, fontweight='bold')
        self.ax_grf.grid(True, linestyle='--', alpha=0.5)

        # 2. TRC Heel Height Plot
        self.ax_trc.plot(self.t_trc[mask_t], self.r_heel_y[mask_t], color='darkorange', linewidth=1.5, label='Right Heel Height (RCAL Y)')
        self.ax_trc.plot(self.t_trc[mask_t], self.l_heel_y[mask_t], color='darkgreen', linewidth=1.5, label='Left Heel Height (LCAL Y)')
        self.ax_trc.set_xlabel("Time (s)")
        self.ax_trc.set_ylabel("Heel Height (mm)")
        self.ax_trc.set_title("Heel Marker Trajectory Verification", fontsize=11, fontweight='bold')
        self.ax_trc.grid(True, linestyle='--', alpha=0.5)

        # Overlay Heel Strike Event Markers
        r_window_hs = [t for t in self.r_hs_times if t_start <= t <= t_end]
        l_window_hs = [t for t in self.l_hs_times if t_start <= t <= t_end]

        if len(r_window_hs) > 0:
            r_grf_y = np.interp(r_window_hs, self.t_grf, self.r_fy_filt)
            r_trc_y = np.interp(r_window_hs, self.t_trc, self.r_heel_y)
            self.ax_grf.scatter(r_window_hs, r_grf_y, color='red', marker='o', s=100, label='Right HS', zorder=7)
            self.ax_trc.scatter(r_window_hs, r_trc_y, color='red', marker='o', s=100, label='Right HS', zorder=7)
            for t in r_window_hs:
                self.ax_grf.axvline(t, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                self.ax_trc.axvline(t, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

        if len(l_window_hs) > 0:
            l_grf_y = np.interp(l_window_hs, self.t_grf, self.l_fy_filt)
            l_trc_y = np.interp(l_window_hs, self.t_trc, self.l_heel_y)
            self.ax_grf.scatter(l_window_hs, l_grf_y, color='blue', marker='^', s=100, label='Left HS', zorder=7)
            self.ax_trc.scatter(l_window_hs, l_trc_y, color='blue', marker='^', s=100, label='Left HS', zorder=7)
            for t in l_window_hs:
                self.ax_grf.axvline(t, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
                self.ax_trc.axvline(t, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)

        # Overlay Toe Off Event Markers
        r_window_to = [t for t in self.r_to_times if t_start <= t <= t_end]
        l_window_to = [t for t in self.l_to_times if t_start <= t <= t_end]

        if len(r_window_to) > 0:
            r_to_grf_y = np.interp(r_window_to, self.t_grf, self.r_fy_filt)
            r_to_trc_y = np.interp(r_window_to, self.t_trc, self.r_heel_y)
            self.ax_grf.scatter(r_window_to, r_to_grf_y, color='darkorange', marker='x', s=90, linewidth=2, label='Right TO', zorder=7)
            self.ax_trc.scatter(r_window_to, r_to_trc_y, color='darkorange', marker='x', s=90, linewidth=2, label='Right TO', zorder=7)
            for t in r_window_to:
                self.ax_grf.axvline(t, color='darkorange', linestyle=':', linewidth=1.2, alpha=0.8)
                self.ax_trc.axvline(t, color='darkorange', linestyle=':', linewidth=1.2, alpha=0.8)

        if len(l_window_to) > 0:
            l_to_grf_y = np.interp(l_window_to, self.t_grf, self.l_fy_filt)
            l_to_trc_y = np.interp(l_window_to, self.t_trc, self.l_heel_y)
            self.ax_grf.scatter(l_window_to, l_to_grf_y, color='darkgreen', marker='x', s=90, linewidth=2, label='Left TO', zorder=7)
            self.ax_trc.scatter(l_window_to, l_to_trc_y, color='darkgreen', marker='x', s=90, linewidth=2, label='Left TO', zorder=7)
            for t in l_window_to:
                self.ax_grf.axvline(t, color='darkgreen', linestyle=':', linewidth=1.2, alpha=0.8)
                self.ax_trc.axvline(t, color='darkgreen', linestyle=':', linewidth=1.2, alpha=0.8)

        self.ax_grf.set_xlim(t_start, t_end)
        self.ax_trc.set_xlim(t_start, t_end)

        self.ax_grf.legend(loc='upper right', fontsize=8)
        self.ax_trc.legend(loc='upper right', fontsize=8)

        self.canvas.draw()

    def _on_finish(self):
        try:
            self.result_speed = float(self.speed_var.get())
            self.result_slope = float(self.slope_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for Speed and Slope.")
            return

        self.finished = True
        self._close_gui()

    def _on_close_window(self):
        self._close_gui()

    def _close_gui(self):
        try:
            if hasattr(self, 'fig') and self.fig is not None:
                plt.close(self.fig)
            plt.close('all')
        except Exception as e:
            print(f"Warning closing plot figure: {e}")

        try:
            if hasattr(self, 'root') and self.root is not None:
                self.root.quit()
                self.root.destroy()
        except Exception as e:
            print(f"Warning destroying Tk window: {e}")


def run_interactive_selector(trial, initial_r_hs=None, initial_l_hs=None,
                             initial_r_to=None, initial_l_to=None,
                             speed=0.0, slope=0.0, postproc_version='v2'):
    """
    Launch the GaitEventSelectorGUI in a modal Tk window.

    Returns:
        tuple: (r_hs_times, l_hs_times, r_to_times, l_to_times, speed, slope)
    """
    root = tk.Tk()
    app = GaitEventSelectorGUI(root, trial=trial, initial_r_hs=initial_r_hs,
                               initial_l_hs=initial_l_hs, initial_r_to=initial_r_to,
                               initial_l_to=initial_l_to, speed=speed,
                               slope=slope, postproc_version=postproc_version)
    
    root.protocol("WM_DELETE_WINDOW", app._on_close_window)
    root.mainloop()

    # Safely copy values after mainloop unblocks
    r_hs = list(app.r_hs_times)
    l_hs = list(app.l_hs_times)
    r_to = list(app.r_to_times)
    l_to = list(app.l_to_times)
    spd = float(app.result_speed)
    slp = float(app.result_slope)

    try:
        root.destroy()
    except Exception:
        pass

    return r_hs, l_hs, r_to, l_to, spd, slp
