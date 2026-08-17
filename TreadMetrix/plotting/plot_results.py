import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define constants
import sys
DEFAULT_DATA_ROOT = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\K4"
DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_ROOT
STIFFNESS = os.path.basename(DATA_ROOT.rstrip('\\/'))

# Stance phase percentages derived dynamically from baseline-corrected force data
STANCE_PERCENTAGES = {}

# Default stance percentages if a trial is not in the list
DEFAULT_STANCE = {'Right': 60.0, 'Left': 60.0}

# Curated colors for comparison plots (used for different speeds or slopes)
LINE_COLORS = [
    '#1565c0',  # Royal Blue
    '#e65100',  # Deep Orange
    '#2e7d32',  # Forest Green
    '#c62828',  # Dark Red
    '#6a1b9a',  # Purple
    '#4e342e',  # Dark Brown
    '#00838f',  # Teal
    '#37474f',  # Blue Gray
]

def parse_condition_label(folder_name):
    """
    Parse speed and slope from folder name (e.g. 'Speed0_5slope3' -> '0.5 mph, 3% slope').
    """
    match = re.match(r"Speed([\d_]+)slope([\d_]+)", folder_name)
    if match:
        speed_str = match.group(1).replace('_', '.')
        slope_str = match.group(2).replace('_', '.')
        return f"{speed_str} mph", f"{slope_str}% slope"
    return folder_name, ""

def parse_condition_details(folder_name):
    """
    Parse speed and slope values as floats from condition folder name.
    """
    match = re.match(r"Speed([\d_]+)slope([\d_]+)", folder_name)
    if match:
        speed = float(match.group(1).replace('_', '.'))
        slope = float(match.group(2).replace('_', '.'))
        return speed, slope
    return None, None

def read_mot_file(filepath):
    """
    Read an OpenSim .mot file, skipping the header lines dynamically.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        header_len = 0
        for idx, line in enumerate(lines):
            if 'endheader' in line:
                header_len = idx + 1
                break
        
        if header_len == 0:
            df = pd.read_csv(filepath, sep=r'\s+')
        else:
            df = pd.read_csv(filepath, sep=r'\s+', skiprows=header_len)
        return df
    except Exception as e:
        print(f"Error reading .mot file {os.path.basename(filepath)}: {e}")
        return None

def calculate_stance_percentages(data_root):
    """
    Dynamically calculate stance phase percentages from the segmented force files.
    """
    stance_percentages = {}
    if not os.path.exists(data_root):
        return stance_percentages
        
    trials = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for trial in trials:
        trial_dir = os.path.join(data_root, trial)
        stance_percentages[trial] = {}
        for side in ["Right", "Left"]:
            segmented_dir = os.path.join(trial_dir, "segmented", side)
            if not os.path.exists(segmented_dir):
                stance_percentages[trial][side] = 60.0  # fallback default
                continue
                
            mot_files = glob.glob(os.path.join(segmented_dir, "*.mot"))
            if not mot_files:
                stance_percentages[trial][side] = 60.0  # fallback default
                continue
                
            percentages = []
            for file_path in mot_files:
                try:
                    if os.path.isdir(file_path):
                        continue
                    df = read_mot_file(file_path)
                    if df is None or len(df) == 0:
                        continue
                    
                    # Find vertical force column (must contain 'vy')
                    col_name = None
                    for col in df.columns:
                        if 'vy' in col.lower():
                            if side == 'Right' and ('force2' in col.lower() or 'force_r' in col.lower() or 'vy2' in col.lower() or '2_vy' in col.lower()):
                                col_name = col
                                break
                            elif side == 'Left' and ('force1' in col.lower() or 'force_l' in col.lower() or 'vy1' in col.lower() or '1_vy' in col.lower()):
                                col_name = col
                                break
                    
                    if col_name is None:
                        # Fallback to finding any column with 'vy'
                        for col in df.columns:
                            if 'vy' in col.lower():
                                col_name = col
                                break
                                
                    if col_name is not None and col_name in df.columns:
                        fy = df[col_name].values
                        above_threshold = np.where(fy > 10.0)[0]
                        if len(above_threshold) > 0:
                            last_stance_idx = above_threshold[-1]
                            pct = (last_stance_idx + 1) / len(fy) * 100
                            percentages.append(pct)
                except Exception as e:
                    print(f"Error calculating stance percentage for {os.path.basename(file_path)}: {e}")
            
            if percentages:
                mean_pct = np.mean(percentages)
                stance_percentages[trial][side] = round(mean_pct, 2)
            else:
                stance_percentages[trial][side] = 60.0  # fallback default
                
    return stance_percentages

def time_normalize(data_series, target_len=101):
    """
    Normalize a data series to exactly 101 points (0% to 100% of gait cycle).
    Uses cubic spline interpolation, falling back to linear if points are too few.
    """
    x_orig = np.linspace(0, 100, len(data_series))
    x_target = np.linspace(0, 100, target_len)
    
    if len(data_series) < 4:
        return np.interp(x_target, x_orig, data_series)
    else:
        f = interp1d(x_orig, data_series, kind='cubic', fill_value="extrapolate")
        return f(x_target)

def load_cycle_data(folder_path, file_extension, column_selector, is_mot=True):
    """
    Load and normalize all cycle files in a given directory.
    """
    if not os.path.exists(folder_path):
        return []
    
    search_pattern = os.path.join(folder_path, f"*.{file_extension}")
    files = glob.glob(search_pattern)
    files = [f for f in files if not os.path.isdir(f) and not f.endswith("_Ankle_Power.png")]
    
    cycles_data = []
    for f in files:
        if is_mot:
            df = read_mot_file(f)
        else:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"Error reading CSV {os.path.basename(f)}: {e}")
                df = None
                
        if df is not None:
            matched_col = None
            for col in df.columns:
                if column_selector(col):
                    matched_col = col
                    break
            
            if matched_col is not None:
                series = df[matched_col].values
                normalized = time_normalize(series)
                cycles_data.append(normalized)
                
    return cycles_data

def apply_plot_style(ax, title, ylabel, xlabel="Gait Cycle (%)"):
    """
    Apply a consistent premium style to a matplotlib axes object.
    """
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
    ax.set_xlim(0, 100)
    ax.grid(True, color='#e0e0e0', linestyle='--', linewidth=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.tick_params(colors='#444444', labelsize=9)

def draw_stance_phase(ax, stance_pct, y_min, y_max, border_color='#d32f2f', label_offset=0.93, label_text='STANCE'):
    """
    Draw a vertical dashed line at toe-off and add shaded stance/swing indicators.
    """
    ax.axvline(x=stance_pct, color=border_color, linestyle='--', linewidth=1.2)
    ax.axvspan(0, stance_pct, color='#f5f5f5', alpha=0.6, zorder=0)
    
    y_text_pos = y_min + (y_max - y_min) * label_offset
    ax.text(stance_pct / 2, y_text_pos, label_text, color='#666666', fontsize=8, fontweight='bold', ha='center', va='center')
    ax.text(stance_pct + (100 - stance_pct) / 2, y_text_pos, 'SWING', color='#666666', fontsize=8, fontweight='bold', ha='center', va='center')

def plot_single_condition(trial_name, condition, trial_data, output_dir, variables, sides, colors, stance_info):
    """
    Generate all individual and combined plots for a single speed/slope condition.
    """
    speed_label, slope_label = parse_condition_label(condition)
    condition_title = f"{speed_label} | {slope_label}" if slope_label else speed_label
    
    # 1. Individual plots (Raw cycles + Mean + SD)
    for var_name, var_info in variables.items():
        for side in sides:
            cycles = trial_data[condition][side][var_name]['raw']
            if len(cycles) == 0:
                continue
                
            mean_line = trial_data[condition][side][var_name]['mean']
            std_line = trial_data[condition][side][var_name]['std']
            percent = np.linspace(0, 100, 101)
            
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            
            # Plot individual cycles
            for cycle in cycles:
                plt.plot(percent, cycle, color=colors[side]['light'], alpha=0.3, linewidth=0.8)
                
            # Plot mean and SD band
            plt.plot(percent, mean_line, color=colors[side]['primary'], linewidth=2.5, label='Mean')
            plt.fill_between(percent, mean_line - std_line, mean_line + std_line, 
                             color=colors[side]['primary'], alpha=0.15, label='±1 SD')
            
            title_text = f"{side} {var_info['title']}"
            apply_plot_style(ax, title_text, var_info['ylabel'])
            
            y_min, y_max = ax.get_ylim()
            draw_stance_phase(ax, stance_info[side], y_min, y_max)
            ax.set_ylim(y_min, y_max)
            
            plt.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
            
            plt.text(0.98, 0.02, f"Trial: {trial_name}\nCond: {condition_title}\nCycles: {len(cycles)}", 
                     transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#666666',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            filename = f"{trial_name}_{condition}_{side}_ankle_{var_name.lower()}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close()
            
    # 2. Combined grid plot (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
    
    for row_idx, side in enumerate(sides):
        for col_idx, (var_name, var_info) in enumerate(variables.items()):
            ax = axes[row_idx, col_idx]
            cycles = trial_data[condition][side][var_name]['raw']
            
            if len(cycles) > 0:
                mean_line = trial_data[condition][side][var_name]['mean']
                std_line = trial_data[condition][side][var_name]['std']
                percent = np.linspace(0, 100, 101)
                
                for cycle in cycles:
                    ax.plot(percent, cycle, color=colors[side]['light'], alpha=0.25, linewidth=0.6)
                    
                ax.plot(percent, mean_line, color=colors[side]['primary'], linewidth=2.2, label='Mean')
                ax.fill_between(percent, mean_line - std_line, mean_line + std_line, 
                                 color=colors[side]['primary'], alpha=0.15, label='±1 SD')
                
                y_min, y_max = ax.get_ylim()
                draw_stance_phase(ax, stance_info[side], y_min, y_max)
                ax.set_ylim(y_min, y_max)
                
            title_text = f"{side} {var_info['title']}"
            apply_plot_style(ax, title_text, var_info['ylabel'])
            if row_idx == 0:
                ax.set_xlabel("")
                
    # Sync Y-limits for side-by-side comparison
    for col_idx, var_name in enumerate(variables):
        left_lim = axes[0, col_idx].get_ylim()
        right_lim = axes[1, col_idx].get_ylim()
        combined_min = min(left_lim[0], right_lim[0])
        combined_max = max(left_lim[1], right_lim[1])
        
        span = combined_max - combined_min
        combined_min -= span * 0.05
        combined_max += span * 0.05
        
        axes[0, col_idx].set_ylim(combined_min, combined_max)
        axes[1, col_idx].set_ylim(combined_min, combined_max)
        
        # Redraw text at new scale
        for row_idx, side in enumerate(sides):
            ax = axes[row_idx, col_idx]
            for t in [t for t in ax.texts]:
                t.remove()
            draw_stance_phase(ax, stance_info[side], combined_min, combined_max)
            ax.set_ylim(combined_min, combined_max)

    fig.suptitle(f"Ankle Biomechanics Analysis — Trial: {trial_name}\nCondition: {condition_title}", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    summary_filename = f"{trial_name}_{condition}_ankle_summary.png"
    summary_filepath = os.path.join(output_dir, summary_filename)
    plt.savefig(summary_filepath, dpi=200, bbox_inches='tight')
    plt.close()

def plot_speed_comparison_group(trial_name, slope_val, speed_cond_list, trial_data, output_dir, variables, sides, stance_info):
    """
    Generate speed-comparison plots where mean curves of all speeds at a fixed slope are overlaid.
    """
    print(f"  Generating speed comparison for slope {slope_val}% (Speeds: {[s[0] for s in speed_cond_list]} mph)...")
    
    slope_title = f"{slope_val}% slope"
    percent = np.linspace(0, 100, 101)
    
    # 1. Combined speed comparison grid plot (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
    
    has_valid_data = False
    
    for row_idx, side in enumerate(sides):
        for col_idx, (var_name, var_info) in enumerate(variables.items()):
            ax = axes[row_idx, col_idx]
            
            # Plot the mean curve for each speed
            for speed_idx, (speed_val, cond_name) in enumerate(speed_cond_list):
                if cond_name in trial_data and side in trial_data[cond_name]:
                    mean_curve = trial_data[cond_name][side][var_name]['mean']
                    if mean_curve is not None:
                        color = LINE_COLORS[speed_idx % len(LINE_COLORS)]
                        ax.plot(percent, mean_curve, color=color, linewidth=2.0, label=f"{speed_val} mph")
                        has_valid_data = True
            
            # Title & formatting
            title_text = f"{side} {var_info['title']}"
            apply_plot_style(ax, title_text, var_info['ylabel'])
            if row_idx == 0:
                ax.set_xlabel("")
                
            # Draw stance phase indicators
            y_min, y_max = ax.get_ylim()
            draw_stance_phase(ax, stance_info[side], y_min, y_max)
            ax.set_ylim(y_min, y_max)
            
            # Show legend on the ankle power subplot
            if col_idx == 2:
                ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', fontsize=8)
                
    if not has_valid_data:
        plt.close()
        return

    # Sync Y-limits for Left/Right variables comparison
    for col_idx, var_name in enumerate(variables):
        left_lim = axes[0, col_idx].get_ylim()
        right_lim = axes[1, col_idx].get_ylim()
        combined_min = min(left_lim[0], right_lim[0])
        combined_max = max(left_lim[1], right_lim[1])
        
        span = combined_max - combined_min
        combined_min -= span * 0.05
        combined_max += span * 0.05
        
        axes[0, col_idx].set_ylim(combined_min, combined_max)
        axes[1, col_idx].set_ylim(combined_min, combined_max)
        
        # Redraw text at synchronized scale
        for row_idx, side in enumerate(sides):
            ax = axes[row_idx, col_idx]
            for t in [t for t in ax.texts]:
                t.remove()
            draw_stance_phase(ax, stance_info[side], combined_min, combined_max)
            ax.set_ylim(combined_min, combined_max)

    fig.suptitle(f"Ankle Biomechanics Speed Effect — Trial: {trial_name}\nFixed Slope: {slope_title} | Stiffness: {STIFFNESS}", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    summary_filename = f"{trial_name}_slope{str(slope_val).replace('.', '_')}_speed_comparison.png"
    summary_filepath = os.path.join(output_dir, summary_filename)
    plt.savefig(summary_filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Individual speed comparison plots
    for var_name, var_info in variables.items():
        for side in sides:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            
            valid_curves = 0
            for speed_idx, (speed_val, cond_name) in enumerate(speed_cond_list):
                if cond_name in trial_data and side in trial_data[cond_name]:
                    mean_curve = trial_data[cond_name][side][var_name]['mean']
                    if mean_curve is not None:
                        color = LINE_COLORS[speed_idx % len(LINE_COLORS)]
                        plt.plot(percent, mean_curve, color=color, linewidth=2.2, label=f"{speed_val} mph")
                        valid_curves += 1
            
            if valid_curves == 0:
                plt.close()
                continue
                
            title_text = f"{side} {var_info['title']} - Speed Comparison"
            apply_plot_style(ax, title_text, var_info['ylabel'])
            
            y_min, y_max = ax.get_ylim()
            draw_stance_phase(ax, stance_info[side], y_min, y_max)
            ax.set_ylim(y_min, y_max)
            
            plt.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
            
            plt.text(0.98, 0.02, f"Trial: {trial_name}\nSlope: {slope_title}\nStiffness: {STIFFNESS}", 
                     transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#666666',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            filename = f"{trial_name}_slope{str(slope_val).replace('.', '_')}_{side}_ankle_{var_name.lower()}_speed_comparison.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close()

def plot_slope_comparison_group(speed_val, slope_cond_list, all_data, output_dir, variables, sides):
    """
    Generate slope-comparison plots where mean curves of all incline slopes at a fixed speed are overlaid.
    """
    print(f"  Generating slope comparison for speed {speed_val} mph (Slopes: {[s[0] for s in slope_cond_list]}%)...")
    
    speed_title = f"{speed_val} mph"
    percent = np.linspace(0, 100, 101)
    
    # 1. Combined slope comparison grid plot (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
    
    has_valid_data = False
    
    for row_idx, side in enumerate(sides):
        for col_idx, (var_name, var_info) in enumerate(variables.items()):
            ax = axes[row_idx, col_idx]
            
            # Plot the mean curve for each slope and draw vertical toe-off lines
            for slope_idx, (slope_val, trial_name, cond_name) in enumerate(slope_cond_list):
                if trial_name in all_data and cond_name in all_data[trial_name] and side in all_data[trial_name][cond_name]:
                    mean_curve = all_data[trial_name][cond_name][side][var_name]['mean']
                    if mean_curve is not None:
                        color = LINE_COLORS[slope_idx % len(LINE_COLORS)]
                        ax.plot(percent, mean_curve, color=color, linewidth=2.0, label=f"{slope_val}% slope")
                        has_valid_data = True
                        
                        # Draw matching-color vertical toe-off line for this slope
                        stance_pct = STANCE_PERCENTAGES.get(trial_name, DEFAULT_STANCE)[side]
                        ax.axvline(x=stance_pct, color=color, linestyle='--', linewidth=1.0, alpha=0.7)
            
            # Title & formatting
            title_text = f"{side} {var_info['title']}"
            apply_plot_style(ax, title_text, var_info['ylabel'])
            if row_idx == 0:
                ax.set_xlabel("")
                
            # Add general stance/swing background shading based on average stance phase (e.g. ~69% for R, ~68% for L)
            avg_stance = np.mean([STANCE_PERCENTAGES.get(s[1], DEFAULT_STANCE)[side] for s in slope_cond_list])
            y_min, y_max = ax.get_ylim()
            ax.axvspan(0, avg_stance, color='#f5f5f5', alpha=0.4, zorder=0)
            y_text_pos = y_min + (y_max - y_min) * 0.93
            ax.text(avg_stance / 2, y_text_pos, 'STANCE', color='#777777', fontsize=8, fontweight='bold', ha='center', va='center')
            ax.text(avg_stance + (100 - avg_stance) / 2, y_text_pos, 'SWING', color='#777777', fontsize=8, fontweight='bold', ha='center', va='center')
            ax.set_ylim(y_min, y_max)
            
            if col_idx == 2:
                ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', fontsize=8)
                
    if not has_valid_data:
        plt.close()
        return

    # Sync Y-limits for Left/Right variables comparison
    for col_idx, var_name in enumerate(variables):
        left_lim = axes[0, col_idx].get_ylim()
        right_lim = axes[1, col_idx].get_ylim()
        combined_min = min(left_lim[0], right_lim[0])
        combined_max = max(left_lim[1], right_lim[1])
        
        span = combined_max - combined_min
        combined_min -= span * 0.05
        combined_max += span * 0.05
        
        axes[0, col_idx].set_ylim(combined_min, combined_max)
        axes[1, col_idx].set_ylim(combined_min, combined_max)
        
        # Redraw text at synchronized scale
        for row_idx, side in enumerate(sides):
            ax = axes[row_idx, col_idx]
            # Remove old text labels
            texts = [t for t in ax.texts]
            for t in texts:
                t.remove()
            # Redraw
            avg_stance = np.mean([STANCE_PERCENTAGES.get(s[1], DEFAULT_STANCE)[side] for s in slope_cond_list])
            y_text_pos = combined_min + (combined_max - combined_min) * 0.93
            ax.text(avg_stance / 2, y_text_pos, 'STANCE', color='#777777', fontsize=8, fontweight='bold', ha='center', va='center')
            ax.text(avg_stance + (100 - avg_stance) / 2, y_text_pos, 'SWING', color='#777777', fontsize=8, fontweight='bold', ha='center', va='center')
            ax.set_ylim(combined_min, combined_max)

    fig.suptitle(f"Ankle Biomechanics Slope Effect — Fixed Speed: {speed_title}\nStiffness: {STIFFNESS}", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    summary_filename = f"{STIFFNESS}_speed{str(speed_val).replace('.', '_')}_slope_comparison.png"
    summary_filepath = os.path.join(output_dir, summary_filename)
    plt.savefig(summary_filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved combined slope comparison: {summary_filename}")
    
    # 2. Individual slope comparison plots
    for var_name, var_info in variables.items():
        for side in sides:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            
            valid_curves = 0
            for slope_idx, (slope_val, trial_name, cond_name) in enumerate(slope_cond_list):
                if trial_name in all_data and cond_name in all_data[trial_name] and side in all_data[trial_name][cond_name]:
                    mean_curve = all_data[trial_name][cond_name][side][var_name]['mean']
                    if mean_curve is not None:
                        color = LINE_COLORS[slope_idx % len(LINE_COLORS)]
                        plt.plot(percent, mean_curve, color=color, linewidth=2.2, label=f"{slope_val}% slope")
                        valid_curves += 1
                        
                        # Draw matching-color vertical toe-off line
                        stance_pct = STANCE_PERCENTAGES.get(trial_name, DEFAULT_STANCE)[side]
                        ax.axvline(x=stance_pct, color=color, linestyle='--', linewidth=1.0, alpha=0.7)
            
            if valid_curves == 0:
                plt.close()
                continue
                
            title_text = f"{side} {var_info['title']} - Slope Comparison"
            apply_plot_style(ax, title_text, var_info['ylabel'])
            
            y_min, y_max = ax.get_ylim()
            avg_stance = np.mean([STANCE_PERCENTAGES.get(s[1], DEFAULT_STANCE)[side] for s in slope_cond_list])
            ax.axvspan(0, avg_stance, color='#f5f5f5', alpha=0.4, zorder=0)
            y_text_pos = y_min + (y_max - y_min) * 0.93
            ax.text(avg_stance / 2, y_text_pos, 'STANCE', color='#777777', fontsize=8, fontweight='bold', ha='center', va='center')
            ax.text(avg_stance + (100 - avg_stance) / 2, y_text_pos, 'SWING', color='#777777', fontsize=8, fontweight='bold', ha='center', va='center')
            ax.set_ylim(y_min, y_max)
            
            plt.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
            
            plt.text(0.98, 0.02, f"Fixed Speed: {speed_title}\nStiffness: {STIFFNESS}", 
                     transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#666666',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            filename = f"{STIFFNESS}_speed{str(speed_val).replace('.', '_')}_{side}_ankle_{var_name.lower()}_slope_comparison.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close()

def main():
    print("=" * 60)
    print(f"Running Results Plotting Pipeline for {STIFFNESS} Trials")
    print("=" * 60)
    
    # Calculate stance percentages dynamically
    STANCE_PERCENTAGES.update(calculate_stance_percentages(DATA_ROOT))
    print("Dynamically calculated Stance Percentages:")
    for trial, sides_pct in STANCE_PERCENTAGES.items():
        print(f"  {trial}: {sides_pct}")
    
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root directory not found at {DATA_ROOT}")
        return
        
    trials = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    variables = {
        'Angle': {
            'subfolder': 'ik_results',
            'ext': 'mot',
            'is_mot': True,
            'col_selector': lambda col, side: col.lower() == f'ankle_angle_{side[0].lower()}',
            'ylabel': 'Angle (deg)',
            'title': 'Ankle Angle'
        },
        'Moment': {
            'subfolder': 'id_results',
            'ext': 'mot',
            'is_mot': True,
            'col_selector': lambda col, side: col.lower() == f'ankle_angle_{side[0].lower()}_moment',
            'ylabel': 'Moment (N-m)',
            'title': 'Ankle Moment'
        },
        'Power': {
            'subfolder': 'power_filtered',
            'ext': 'csv',
            'is_mot': False,
            'col_selector': lambda col, side: col.lower() == f'ankle_angle_{side[0].lower()}_power',
            'ylabel': 'Power (W)',
            'title': 'Ankle Power'
        }
    }
    
    sides = ['Left', 'Right']
    
    colors = {
        'Right': {
            'primary': '#1565c0',  # Royal Blue
            'light': '#90caf9',    # Light Blue
        },
        'Left': {
            'primary': '#2e7d32',  # Forest Green
            'light': '#a5d6a7',    # Light Green
        }
    }
    
    # Global cache to load all trial data
    all_data = {}
    
    # 1. Load and cache all trial data
    print("Loading and normalizing all gait cycle data...")
    for trial in trials:
        trial_dir = os.path.join(DATA_ROOT, trial)
        ik_dir = os.path.join(trial_dir, "ik_results")
        
        if not os.path.exists(ik_dir):
            continue
            
        all_data[trial] = {}
        conditions = [c for c in os.listdir(ik_dir) if os.path.isdir(os.path.join(ik_dir, c))]
        
        for cond in conditions:
            all_data[trial][cond] = {}
            for side in sides:
                all_data[trial][cond][side] = {}
                for var_name, var_info in variables.items():
                    folder_path = os.path.join(trial_dir, var_info['subfolder'], cond, side)
                    col_sel = lambda col, s=side: var_info['col_selector'](col, s)
                    cycles = load_cycle_data(folder_path, var_info['ext'], col_sel, var_info['is_mot'])
                    cycles_arr = np.array(cycles)
                    
                    if len(cycles_arr) > 0:
                        mean_data = np.mean(cycles_arr, axis=0)
                        std_data = np.std(cycles_arr, axis=0)
                    else:
                        mean_data = None
                        std_data = None
                        
                    all_data[trial][cond][side][var_name] = {
                        'raw': cycles_arr,
                        'mean': mean_data,
                        'std': std_data
                    }
                    
    # 2. Generate Single-Condition and Speed Comparison plots for each trial
    for trial in trials:
        if trial not in all_data:
            continue
            
        trial_dir = os.path.join(DATA_ROOT, trial)
        plots_output_dir = os.path.join(trial_dir, "plots")
        os.makedirs(plots_output_dir, exist_ok=True)
        
        print(f"\nProcessing trial: {trial}")
        conditions = list(all_data[trial].keys())
        stance_info = STANCE_PERCENTAGES.get(trial, DEFAULT_STANCE)
        
        # 2a. Single condition plots
        print("  Generating single condition plots...")
        for cond in conditions:
            total_loaded = sum(len(all_data[trial][cond][s][v]['raw']) for s in sides for v in variables)
            if total_loaded > 0:
                plot_single_condition(trial, cond, all_data[trial], plots_output_dir, variables, sides, colors, stance_info)
                
        # 2b. Speed Effect plots (within trial)
        print("  Grouping by slope for speed effect plots...")
        slope_groups = {}
        for cond in conditions:
            speed, slope = parse_condition_details(cond)
            if speed is not None and slope is not None:
                if slope not in slope_groups:
                    slope_groups[slope] = []
                slope_groups[slope].append((speed, cond))
                
        for slope_val, speed_cond_list in slope_groups.items():
            speed_cond_list.sort(key=lambda x: x[0])
            plot_speed_comparison_group(trial, slope_val, speed_cond_list, all_data[trial], plots_output_dir, variables, sides, stance_info)
            
    # 3. Generate Slope Effect plots (Cross-trial comparisons at fixed speeds)
    print("\nGrouping across trials by speed for slope effect plots...")
    global_plots_dir = os.path.join(DATA_ROOT, "plots")
    os.makedirs(global_plots_dir, exist_ok=True)
    print(f"  Cross-trial slope comparison plots will be saved to: {global_plots_dir}")
    
    speed_groups = {}  # {speed_val: [(slope_val, trial_name, cond_name), ...]}
    for trial in all_data:
        for cond in all_data[trial]:
            speed, slope = parse_condition_details(cond)
            if speed is not None and slope is not None:
                if speed not in speed_groups:
                    speed_groups[speed] = []
                speed_groups[speed].append((slope, trial, cond))
                
    for speed_val, slope_cond_list in speed_groups.items():
        # Sort by slope ascending
        slope_cond_list.sort(key=lambda x: x[0])
        plot_slope_comparison_group(speed_val, slope_cond_list, all_data, global_plots_dir, variables, sides)
        
    # 4. Generate Multi-Stiffness Facet Grid Plots
    parent_dir = os.path.dirname(DATA_ROOT.rstrip('\\/'))
    stiffnesses = ['K3', 'K4', 'K5', 'K6']
    available_stiffnesses = [s for s in stiffnesses if os.path.exists(os.path.join(parent_dir, s))]
    if len(available_stiffnesses) > 1 and "--no-facet" not in sys.argv:
        print("\n" + "=" * 60)
        print("Generating Multi-Stiffness Facet Grid Plots...")
        print(f"  Stiffnesses found: {available_stiffnesses}")
        print("=" * 60)
        generate_facet_grids(parent_dir, available_stiffnesses, variables, sides, DATA_ROOT)
        
    print("\n" + "=" * 60)
    print("Results Plotting Completed Successfully!")
    print("=" * 60)


def load_all_stiffness_data(parent_dir, stiffnesses, variables, sides):
    """
    Loads all cycle data for all stiffnesses, trials, and conditions.
    Returns:
        dict: nested dictionary structured as:
              data[side][variable][slope][speed][stiffness] = {
                  'raw': list of 101-pt arrays,
                  'mean': 101-pt array or None,
                  'std': 101-pt array or None
              }
    """
    data = {}
    for side in sides:
        data[side] = {}
        for var_name in variables:
            data[side][var_name] = {}
            
    for stiffness in stiffnesses:
        stiffness_dir = os.path.join(parent_dir, stiffness)
        if not os.path.exists(stiffness_dir):
            continue
        
        # List trials inside stiffness folder
        trials = [t for t in os.listdir(stiffness_dir) if os.path.isdir(os.path.join(stiffness_dir, t)) and t != 'plots']
        
        for trial in trials:
            trial_dir = os.path.join(stiffness_dir, trial)
            for var_name, var_info in variables.items():
                var_dir = os.path.join(trial_dir, var_info['subfolder'])
                if not os.path.exists(var_dir):
                    continue
                
                # List conditions (e.g. Speed0_5slope3)
                conditions = [c for c in os.listdir(var_dir) if os.path.isdir(os.path.join(var_dir, c))]
                for cond in conditions:
                    speed, slope = parse_condition_details(cond)
                    if speed is None or slope is None:
                        continue
                    
                    for side in sides:
                        folder_path = os.path.join(var_dir, cond, side)
                        col_sel = lambda col, s=side: var_info['col_selector'](col, s)
                        cycles = load_cycle_data(folder_path, var_info['ext'], col_sel, var_info['is_mot'])
                        
                        if len(cycles) > 0:
                            # Ensure dictionaries are initialized
                            if slope not in data[side][var_name]:
                                data[side][var_name][slope] = {}
                            if speed not in data[side][var_name][slope]:
                                data[side][var_name][slope][speed] = {}
                            if stiffness not in data[side][var_name][slope][speed]:
                                data[side][var_name][slope][speed][stiffness] = []
                            
                            data[side][var_name][slope][speed][stiffness].extend(cycles)
                            
    # Now compute mean and std for each leaf node
    for side in sides:
        for var_name in data[side]:
            for slope in data[side][var_name]:
                for speed in data[side][var_name][slope]:
                    for stiffness in data[side][var_name][slope][speed]:
                        cycles_list = data[side][var_name][slope][speed][stiffness]
                        cycles_arr = np.array(cycles_list)
                        data[side][var_name][slope][speed][stiffness] = {
                            'raw': cycles_arr,
                            'mean': np.mean(cycles_arr, axis=0) if len(cycles_arr) > 0 else None,
                            'std': np.std(cycles_arr, axis=0) if len(cycles_arr) > 0 else None
                        }
                        
    return data


def generate_facet_grids(parent_dir, stiffnesses, variables, sides, output_base):
    """
    Generates facet grids of stiffness vs speed for each slope, side, and variable.
    """
    print("Loading all stiffness data for facet grids...")
    data = load_all_stiffness_data(parent_dir, stiffnesses, variables, sides)
    
    # Custom colors for stiffnesses
    stiffness_colors = {
        'K3': '#1565c0',  # Royal Blue
        'K4': '#e65100',  # Deep Orange
        'K5': '#2e7d32',  # Forest Green
        'K6': '#c62828'   # Dark Red
    }
    
    # Output directories
    out_dirs = [
        os.path.join(output_base, "plots", "facet_grids"),
        os.path.join(parent_dir, "plots", "facet_grids")
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    percent = np.linspace(0, 100, 101)
    
    for side in sides:
        for var_name, var_info in variables.items():
            if side not in data or var_name not in data[side]:
                continue
            slopes_with_data = sorted(list(data[side][var_name].keys()))
            
            for slope in slopes_with_data:
                # Find speeds that have data for this slope across any stiffness
                speeds = sorted(list(data[side][var_name][slope].keys()))
                if not speeds:
                    continue
                
                nrows = len(stiffnesses)
                ncols = len(speeds)
                
                fig, axes = plt.subplots(
                    nrows, ncols, 
                    sharex=True, sharey=True, 
                    figsize=(ncols * 3.2 + 1, nrows * 2.6 + 1.2),
                    squeeze=False
                )
                
                has_any_data = False
                
                for row_idx, stiffness in enumerate(stiffnesses):
                    for col_idx, speed in enumerate(speeds):
                        ax = axes[row_idx, col_idx]
                        
                        # Apply style
                        apply_plot_style(ax, "", "")
                        
                        # Get data
                        stiff_data = data[side][var_name][slope].get(speed, {}).get(stiffness, None)
                        
                        if stiff_data is not None and len(stiff_data['raw']) > 0:
                            has_any_data = True
                            raw_cycles = stiff_data['raw']
                            mean_curve = stiff_data['mean']
                            
                            # Plot individual cycles in light grey
                            for cycle in raw_cycles:
                                ax.plot(percent, cycle, color='#d3d3d3', alpha=0.35, linewidth=0.6)
                            
                            # Plot mean curve
                            color = stiffness_colors.get(stiffness, '#000000')
                            ax.plot(percent, mean_curve, color=color, linewidth=2.0, label=f"{stiffness}")
                            
                            # Reference line for toe-off (e.g. 60%)
                            ax.axvline(x=60.0, color='#999999', linestyle='--', linewidth=0.8, alpha=0.6)
                            
                            # Label count of cycles
                            ax.text(0.95, 0.05, f"N={len(raw_cycles)}", transform=ax.transAxes,
                                    ha='right', va='bottom', fontsize=8, color='#888888')
                        else:
                            # Print "No Data" label in the middle of empty plots
                            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                                    ha='center', va='center', fontsize=10, color='#aaaaaa', style='italic')
                            # Set background to very faint gray
                            ax.set_facecolor('#fafafa')
                        
                        # Column titles on the top row
                        if row_idx == 0:
                            ax.set_title(f"{speed} mph", fontsize=11, fontweight='bold', pad=8)
                        
                        # Row labels on the right column
                        if col_idx == ncols - 1:
                            # We place a text box on the right of the plot
                            ax.text(1.05, 0.5, stiffness, transform=ax.transAxes,
                                    ha='left', va='center', fontsize=12, fontweight='bold',
                                    color=stiffness_colors.get(stiffness, '#000000'), rotation=270)
                
                if not has_any_data:
                    plt.close(fig)
                    continue
                
                # Global labels
                # Title
                fig.suptitle(
                    f"AFO Ankle Biomechanics — {side} {var_info['title']} Facet Grid\n"
                    f"Slope: {slope}% | Rows = Stiffness, Columns = Speed",
                    fontsize=14, fontweight='bold', y=0.98
                )
                
                # Common Y label (for the first column)
                for row_idx in range(nrows):
                    axes[row_idx, 0].set_ylabel(var_info['ylabel'], fontsize=10, labelpad=5)
                    
                # Common X label (for the bottom row)
                for col_idx in range(ncols):
                    axes[nrows-1, col_idx].set_xlabel("Gait Cycle (%)", fontsize=10, labelpad=5)
                
                plt.tight_layout()
                fig.subplots_adjust(top=0.88, right=0.93)
                
                # Save figures
                filename = f"facet_{side.lower()}_ankle_{var_name.lower()}_slope{str(slope).replace('.', '_')}.png"
                for d in out_dirs:
                    filepath = os.path.join(d, filename)
                    plt.savefig(filepath, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved facet grid: {filename}")


if __name__ == '__main__':
    main()
