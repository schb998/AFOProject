import os
import glob
import re
import argparse
import webbrowser
import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def read_mot_fast(file_path):
    """Fast parsing of an OpenSim .mot file using numpy."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        idx = content.find('endheader')
        if idx == -1:
            return None, None
            
        data_str = content[idx + len('endheader'):].strip()
        lines = data_str.splitlines()
        if len(lines) < 2:
            return None, None
            
        header = lines[0].strip().split('\t')
        
        rows = []
        for line in lines[1:]:
            s = line.strip()
            if s:
                parts = s.split('\t')
                if len(parts) == len(header):
                    try:
                        rows.append([float(x) for x in parts])
                    except ValueError:
                        continue
                        
        if not rows:
            return None, None
            
        return header, np.array(rows)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def normalize_time(time_arr, data_arr, num_points=101):
    """Normalizes time series to 0-100% Gait Cycle with specified points."""
    if len(time_arr) < 2 or np.isnan(data_arr).all():
        return np.linspace(0, 100, num_points), np.full(num_points, np.nan)
    
    t0, t1 = time_arr[0], time_arr[-1]
    if t1 <= t0:
        return np.linspace(0, 100, num_points), np.full(num_points, np.nan)
        
    t_norm = (time_arr - t0) / (t1 - t0)
    new_t = np.linspace(0, 1, num_points)
    
    # Interpolate
    f = interp1d(t_norm, data_arr, kind='linear', fill_value='extrapolate')
    return new_t * 100, f(new_t)

def parse_cycle_info(file_path, base_dir):
    """Extracts condition, side, cycle number from filepath/filename."""
    rel_path = os.path.relpath(file_path, base_dir)
    parts = rel_path.split(os.sep)
    filename = os.path.basename(file_path)
    
    # Condition: first subdirectory or default
    condition = parts[0] if len(parts) > 1 else "Default"
    
    # Side detection
    if "Left" in parts or "_Left_" in filename or filename.startswith("Left"):
        side = "Left"
    elif "Right" in parts or "_Right_" in filename or filename.startswith("Right"):
        side = "Right"
    else:
        side = "Unknown"
        
    # Cycle number
    m = re.search(r'cycle(\d+)', filename, re.IGNORECASE)
    cycle_num = int(m.group(1)) if m else -1
    
    return {
        'filepath': file_path,
        'filename': filename,
        'rel_path': rel_path,
        'condition': condition,
        'side': side,
        'cycle_num': cycle_num
    }

def main():
    parser = argparse.ArgumentParser(description="Plot Left and Right Ankle Angles interactively.")
    parser.add_argument('--dir', type=str, 
                        default=r"Y:\AFO\Collected Data\P02\K4\K4_Slope_S01\ik_results",
                        help="Directory containing IK mot files")
    parser.add_argument('--out', type=str, default="ankle_angles_interactive.html",
                        help="Output HTML file name")
    parser.add_argument('--no-open', action='store_true', default=False,
                        help="Do not open HTML in browser automatically")
    args = parser.parse_args()

    base_dir = args.dir
    print(f"Scanning directory: {base_dir}")
    
    mot_files = glob.glob(os.path.join(base_dir, '**', '*.mot'), recursive=True)
    print(f"Found {len(mot_files)} .mot files.")

    if not mot_files:
        print("No .mot files found!")
        return

    num_pts = 101
    gait_pct = np.linspace(0, 100, num_pts)
    
    data_left = []
    data_right = []

    for f in mot_files:
        info = parse_cycle_info(f, base_dir)
        header, data_matrix = read_mot_fast(f)
        if header is None or data_matrix is None:
            continue
            
        if 'time' not in header:
            continue

        time_idx = header.index('time')
        time_arr = data_matrix[:, time_idx]

        # Left ankle processing
        if info['side'] == 'Left' or 'ankle_angle_l' in header:
            if 'ankle_angle_l' in header:
                l_idx = header.index('ankle_angle_l')
                l_arr = data_matrix[:, l_idx]
                if not np.isnan(l_arr).all():
                    _, norm_angle = normalize_time(time_arr, l_arr, num_pts)
                    if not np.isnan(norm_angle).all():
                        data_left.append({
                            'info': info,
                            'angle': norm_angle,
                            'min_angle': float(np.min(norm_angle)),
                            'max_angle': float(np.max(norm_angle)),
                            'range_angle': float(np.ptp(norm_angle))
                        })

        # Right ankle processing
        if info['side'] == 'Right' or 'ankle_angle_r' in header:
            if 'ankle_angle_r' in header:
                r_idx = header.index('ankle_angle_r')
                r_arr = data_matrix[:, r_idx]
                if not np.isnan(r_arr).all():
                    _, norm_angle = normalize_time(time_arr, r_arr, num_pts)
                    if not np.isnan(norm_angle).all():
                        data_right.append({
                            'info': info,
                            'angle': norm_angle,
                            'min_angle': float(np.min(norm_angle)),
                            'max_angle': float(np.max(norm_angle)),
                            'range_angle': float(np.ptp(norm_angle))
                        })

    print(f"Successfully processed {len(data_left)} Left cycles and {len(data_right)} Right cycles.")

    # Sort by condition and cycle_num
    data_left.sort(key=lambda x: (x['info']['condition'], x['info']['cycle_num']))
    data_right.sort(key=lambda x: (x['info']['condition'], x['info']['cycle_num']))

    # Create Plotly subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Left Ankle Angle (ankle_angle_l)", "Right Ankle Angle (ankle_angle_r)"),
        horizontal_spacing=0.08
    )

    # Color palettes for different conditions
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173'
    ]

    conditions = sorted(list(set([d['info']['condition'] for d in data_left + data_right])))
    cond_color_map = {cond: color_palette[i % len(color_palette)] for i, cond in enumerate(conditions)}

    # Helper function to add traces
    def add_side_traces(data_list, col_idx, side_name):
        if not data_list:
            return None, None
            
        angles_matrix = np.array([d['angle'] for d in data_list])
        mean_angle = np.nanmean(angles_matrix, axis=0)
        std_angle = np.nanstd(angles_matrix, axis=0)
        
        # 1. Add SD envelope
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([gait_pct, gait_pct[::-1]]),
                y=np.concatenate([mean_angle + std_angle, (mean_angle - std_angle)[::-1]]),
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name=f"{side_name} ±1 SD Band",
                legendgroup=f"{side_name}_stats",
                showlegend=True
            ),
            row=1, col=col_idx
        )

        # 2. Add Mean trace
        fig.add_trace(
            go.Scatter(
                x=gait_pct,
                y=mean_angle,
                mode='lines',
                line=dict(color='black', width=3.5, dash='dash'),
                name=f"{side_name} Mean Angle",
                legendgroup=f"{side_name}_stats",
                hovertemplate=f"<b>{side_name} Mean Angle</b><br>Gait Cycle: %{{x:.1f}}%<br>Angle: %{{y:.2f}}°<extra></extra>",
                showlegend=True
            ),
            row=1, col=col_idx
        )

        # 3. Add Individual Gait Cycle Traces
        for d in data_list:
            cond = d['info']['condition']
            c_num = d['info']['cycle_num']
            fname = d['info']['filename']
            c_color = cond_color_map.get(cond, '#1f77b4')
            
            trace_name = f"{side_name} Cycle #{c_num} ({cond})"
            
            hover_txt = (
                f"<b>{side_name} Gait Cycle #{c_num}</b><br>"
                f"Condition: {cond}<br>"
                f"Gait Cycle: %{{x:.1f}}%<br>"
                f"Ankle Angle: %{{y:.2f}}°<br>"
                f"ROM / Min / Max: {d['range_angle']:.1f}° / {d['min_angle']:.1f}° / {d['max_angle']:.1f}°<br>"
                f"File: {fname}<extra></extra>"
            )
            
            fig.add_trace(
                go.Scatter(
                    x=gait_pct,
                    y=d['angle'],
                    mode='lines',
                    line=dict(width=1.8),
                    name=trace_name,
                    legendgroup=f"{side_name}_{cond}",
                    hovertemplate=hover_txt,
                    opacity=0.75
                ),
                row=1, col=col_idx
            )
            
        return mean_angle, std_angle

    add_side_traces(data_left, 1, "Left")
    add_side_traces(data_right, 2, "Right")

    # Layout styling with buttons for Quick Toggling
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            x=0.5,
            y=1.12,
            xanchor="center",
            yanchor="top",
            buttons=list([
                dict(
                    label="Show All Cycles",
                    method="update",
                    args=[{"visible": [True] * len(fig.data)}]
                ),
                dict(
                    label="Hide All Cycles (Show Mean Only)",
                    method="update",
                    args=[{"visible": [True if "Mean" in t.name or "SD" in t.name else "legendonly" for t in fig.data]}]
                )
            ])
        )
    ]

    fig.update_layout(
        title={
            'text': f"<b>Interactive Ankle Angles Visualization (Left vs Right)</b><br><span style='font-size:13px; color:#555;'>Directory: {base_dir}</span>",
            'x': 0.5,
            'xanchor': 'center'
        },
        template="plotly_white",
        height=750,
        hovermode="closest",
        updatemenus=updatemenus,
        legend=dict(
            title=dict(text="<b>Gait Cycles & Stats</b><br><span style='font-size:10px;font-weight:normal;'>Click to Toggle, Double-Click to Isolate</span>"),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            itemsizing="constant"
        ),
        margin=dict(l=60, r=260, t=110, b=60)
    )

    fig.update_xaxes(title_text="<b>Gait Cycle (%)</b>", range=[0, 100], dtick=10, gridcolor="#e5e5e5", row=1, col=1)
    fig.update_xaxes(title_text="<b>Gait Cycle (%)</b>", range=[0, 100], dtick=10, gridcolor="#e5e5e5", row=1, col=2)
    
    fig.update_yaxes(title_text="<b>Left Ankle Angle (deg)</b>", gridcolor="#e5e5e5", row=1, col=1)
    fig.update_yaxes(title_text="<b>Right Ankle Angle (deg)</b>", gridcolor="#e5e5e5", row=1, col=2)

    # Build full HTML output file
    target_html = os.path.join(base_dir, args.out)
    local_html = os.path.join(os.getcwd(), args.out)

    # Save to both target directory and workspace
    fig.write_html(target_html, include_plotlyjs="cdn")
    fig.write_html(local_html, include_plotlyjs="cdn")

    print(f"\nSaved interactive HTML plots to:")
    print(f"  - Target Dir: {target_html}")
    print(f"  - Local Workspace: {local_html}")

    if not args.no_open:
        try:
            print("Opening interactive dashboard in default web browser...")
            webbrowser.open(target_html, new=2)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")

if __name__ == "__main__":
    main()
