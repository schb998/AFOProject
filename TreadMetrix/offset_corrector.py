import os
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator

class TreadmillOffsetCorrector:
    """
    A class to correct treadmill force plate data based on empty-treadmill
    calibration offsets that vary non-linearly with speed and slope.
    
    This acts as both a Lookup Table (for exact matches) and a 2D Grid 
    Interpolator (Bivariate Spline) for conditions falling between collected data.
    """
    
    def __init__(self, summary_csv_path=r"Y:\AFO_Codes\TreadmillOffset\pooled_treadmill_offsets.csv"):
        """
        Initialize the corrector by loading the summary CSV and building the interpolators.
        """
        if not os.path.exists(summary_csv_path):
            raise FileNotFoundError(f"Offset summary file not found at: {summary_csv_path}")
            
        self.df = pd.read_csv(summary_csv_path)
        
        # We use the median values as they are more robust to isolated noise spikes
        self.force_cols = [
            'ground_force4_vx', 'ground_force4_vy', 'ground_force4_vz',
            'ground_force5_vx', 'ground_force5_vy', 'ground_force5_vz'
        ]
        
        # Input features for interpolation: (Speed, Slope)
        points = self.df[['Speed', 'Slope']].values
        
        # Create 6 independent models (one for each force direction on each plate)
        self.interpolators = {}
        for col in self.force_cols:
            values = self.df[f'{col}_median'].values
            
            # LinearNDInterpolator creates a continuous surface triangulated between the known points.
            # If the exact point exists, it returns the exact value (Lookup Table).
            # If it's between points, it linearly interpolates on the 2D surface.
            self.interpolators[col] = LinearNDInterpolator(points, values)

    def get_offsets(self, speed, slope):
        """
        Retrieve the 6 force offsets for a given speed and slope.
        
        Returns:
            dict: { 'ground_force4_vx': offset_value, ... }
        """
        offsets = {}
        for col in self.force_cols:
            # The interpolator expects an array of points, we give it one point [speed, slope]
            # It returns an array of results, we take the first element [0]
            val = self.interpolators[col](np.array([[speed, slope]]))[0]
            
            # Handle out-of-bounds queries (extrapolation)
            # LinearNDInterpolator returns NaN if requested point is completely outside the convex hull of known points.
            if np.isnan(val):
                # Fallback: Just return the offset of the closest known point
                distances = np.sqrt((self.df['Speed'] - speed)**2 + (self.df['Slope'] - slope)**2)
                closest_idx = distances.idxmin()
                val = self.df.loc[closest_idx, f'{col}_median']
                print(f"Warning: (Speed={speed}, Slope={slope}) is out of calibration bounds. Using nearest neighbor.")
                
            offsets[col] = val
            
        return offsets

    def correct_mot_dataframe(self, df, speed, slope):
        """
        Subtracts the interpolated offsets from the force plate data in the DataFrame.
        
        Args:
            df (pd.DataFrame): The raw .mot file data loaded into a DataFrame.
            speed (float): Treadmill speed in mph.
            slope (float): Treadmill incline percentage.
            
        Returns:
            pd.DataFrame: A new DataFrame with the corrected forces.
        """
        offsets = self.get_offsets(speed, slope)
        
        # Create a copy so we don't accidentally modify the original dataframe in place
        corrected_df = df.copy()
        
        for col in self.force_cols:
            target_col = col
            if col not in corrected_df.columns:
                target_col = col.replace('4', '1').replace('5', '2')
                
            if target_col in corrected_df.columns:
                # Subtract the offset (tare the force plate)
                corrected_df[target_col] = corrected_df[target_col] - offsets[col]
            else:
                print(f"Warning: Column {col} (or mapped {target_col}) not found in the provided DataFrame.")
                
        return corrected_df

    def interactive_correction(self, df, time_col='time', trial_name=''):
        """
        Opens a visual GUI for the user to highlight time windows, then asks
        for treadmill speed (mph) AND slope (%) for each window.
        Offsets are then interpolated from the calibration surface and subtracted
        only within each highlighted window.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import SpanSelector, Button
        import tkinter as tk
        from tkinter import simpledialog

        corrected_df = df.copy()
        selections = []

        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            fig.canvas.manager.set_window_title(f"Treadmill Offset Corrector - Trial: {trial_name}")
        except Exception:
            pass
        
        plot_col = 'ground_force4_vy'
        if plot_col not in df.columns:
            plot_col = 'ground_force1_vy'
            if plot_col not in df.columns:
                # Fallback to the first available non-time column
                plot_col = [c for c in df.columns if c != time_col][0]

        ax.plot(df[time_col], df[plot_col], label=f'Raw {plot_col}', color='steelblue')
        ax.set_title(f"Multi-Speed Trial Corrector - Trial: {trial_name}\nDrag to highlight a steady-speed window. Click 'Finish' when done.", fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.legend(loc='upper left')

        # Tkinter root setup for dialogs
        try:
            if tk._default_root is not None and tk._default_root.winfo_exists():
                root = tk._default_root
            else:
                root = tk.Tk()
                root.withdraw()
        except Exception:
            root = tk.Tk()
            root.withdraw()

        try:
            root.call('wm', 'attributes', '.', '-topmost', True)
        except Exception:
            pass

        def ask_speed_slope(xmin, xmax):
            """Open a compact Tkinter dialog asking for both Speed and Slope."""
            nonlocal root
            try:
                if root is None or not root.winfo_exists():
                    root = tk.Tk()
                    root.withdraw()
            except Exception:
                root = tk.Tk()
                root.withdraw()

            dialog = tk.Toplevel(root)
            dialog.title(f"Window Parameters - Trial: {trial_name}")
            dialog.resizable(False, False)
            dialog.attributes('-topmost', True)
            dialog.grab_set()  # Modal

            tk.Label(dialog,
                     text=f"Selected window:  {xmin:.2f} s  →  {xmax:.2f} s",
                     font=("Helvetica", 10, "bold"), pady=6).grid(row=0, column=0, columnspan=2, padx=14)

            tk.Label(dialog, text="Treadmill Speed (mph):", anchor='w').grid(
                row=1, column=0, sticky='w', padx=14, pady=4)
            speed_var = tk.StringVar(master=dialog, value="0.0")
            speed_entry = tk.Entry(dialog, textvariable=speed_var, width=10)
            speed_entry.grid(row=1, column=1, padx=14, pady=4)
            speed_entry.focus_set()

            tk.Label(dialog, text="Treadmill Slope (%):", anchor='w').grid(
                row=2, column=0, sticky='w', padx=14, pady=4)
            slope_var = tk.StringVar(master=dialog, value="0.0")
            slope_entry = tk.Entry(dialog, textvariable=slope_var, width=10)
            slope_entry.grid(row=2, column=1, padx=14, pady=4)

            result = {}

            def on_ok(event=None):
                try:
                    # Read directly from the Entry widget — more reliable than StringVar
                    # in nested Toplevel windows where StringVar can return stale values.
                    spd = float(speed_entry.get())
                    slp = float(slope_entry.get())
                    if spd < 0 or spd > 15:
                        tk.messagebox.showerror("Invalid", "Speed must be between 0 and 15 mph.", parent=dialog)
                        return
                    result['speed'] = spd
                    result['slope'] = slp
                    dialog.destroy()
                except ValueError:
                    tk.messagebox.showerror("Invalid", "Please enter numeric values for speed and slope.", parent=dialog)

            def on_cancel():
                dialog.destroy()

            btn_frame = tk.Frame(dialog)
            btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
            tk.Button(btn_frame, text="OK",     width=10, command=on_ok).pack(side='left',  padx=6)
            tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side='right', padx=6)

            dialog.bind('<Return>', on_ok)
            root.wait_window(dialog)
            return result

        def onselect(xmin, xmax):
            result = ask_speed_slope(xmin, xmax)
            if result:
                speed = result['speed']
                slope = result['slope']
                selections.append({'tmin': xmin, 'tmax': xmax, 'speed': speed, 'slope': slope})

                # Highlight region on plot
                ax.axvspan(xmin, xmax, color='red', alpha=0.2)

                # Annotate the span with speed AND slope
                y_max = ax.get_ylim()[1]
                label = f"{speed} mph / {slope}%"
                ax.text((xmin + xmax) / 2, y_max * 0.9, label,
                        horizontalalignment='center', color='darkred', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                fig.canvas.draw_idle()

        # Create SpanSelector
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True, interactive=True)

        # Add Finish button
        ax_finish = plt.axes([0.85, 0.02, 0.1, 0.05])
        btn_finish = Button(ax_finish, 'Finish and Apply')

        def finish_clicked(event):
            plt.close(fig)

        btn_finish.on_clicked(finish_clicked)

        print("GUI Opened. Please select the speed windows on the plot and click 'Finish'...")
        plt.show() # Blocks until window is closed

        try:
            if root is not None and root.winfo_exists():
                root.destroy()
        except Exception:
            pass

        if not selections:
            print("No windows selected. Returning original DataFrame.")
            return corrected_df, []

        print(f"\nApplying offsets for {len(selections)} selected windows...")
        for sel in selections:
            tmin, tmax = sel['tmin'], sel['tmax']
            speed = sel['speed']
            slope = sel['slope']
            print(f"  -> Correcting window [{tmin:.2f}s - {tmax:.2f}s] | Speed={speed} mph, Slope={slope}%")

            # Find indices corresponding to this time window
            mask = (corrected_df[time_col] >= tmin) & (corrected_df[time_col] <= tmax)

            # Get offsets for this speed/slope combination
            offsets = self.get_offsets(speed, slope)

            # Apply offsets only to this mask
            for col in self.force_cols:
                target_col = col
                if col not in corrected_df.columns:
                    target_col = col.replace('4', '1').replace('5', '2')

                if target_col in corrected_df.columns:
                    corrected_df.loc[mask, target_col] -= offsets[col]

        print("Piecewise correction complete!")
        # Return both the corrected dataframe AND the selections list so the
        # pipeline can route each gait cycle's output into the correct speed/slope folder.
        return corrected_df, selections
