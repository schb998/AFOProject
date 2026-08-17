"""
marker_event_detection.py
=========================
Hybrid gait event detection for the TreadMetrix pipeline.

Two public functions:

  detect_events_from_markers(trc)
      Zeni (2008) kinematic algorithm: detects heel strikes and toe-offs
      from the relative AP position of heel/toe markers with respect to
      the pelvis centre.  Returns timestamps in seconds.

  reconcile_events(grf_times, marker_times, tolerance_s=0.1)
      Compares one side + event type from both detection sources and
      returns (confirmed_times, suggested_times):
        confirmed  -- GRF and marker both detected within +-tolerance_s
                      (GRF timestamp is kept, higher temporal resolution)
        suggested  -- detected by only one source (GRF-only or marker-only);
                      shown as ghost markers in the GUI but NOT processed
                      unless the user explicitly clicks to promote them.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# Marker name candidates (tried in order, first found is used)
_R_HEEL_CANDIDATES = ['RCAL', 'RHeel', 'RHEEL', 'HEEL_R']
_L_HEEL_CANDIDATES = ['LCAL', 'LHeel', 'LHEEL', 'HEEL_L']
_R_TOE_CANDIDATES  = ['RToe', 'RTOE', 'TOE_R', 'RMT1']
_L_TOE_CANDIDATES  = ['LToe', 'LTOE', 'TOE_L', 'LMT1']
_R_ASI_CANDIDATES  = ['RASI', 'RASIS', 'R_ASIS']
_L_ASI_CANDIDATES  = ['LASI', 'LASIS', 'L_ASIS']


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_marker(marker_dict: dict, candidates: list) -> str | None:
    """Return the first candidate key present in marker_dict, else None."""
    for name in candidates:
        if name in marker_dict:
            return name
    return None


def _get_marker_ap(df, marker_dict: dict, marker_name: str | None, ap_axis_idx: int):
    """Return the AP-axis column values for a marker, or None if unavailable."""
    if marker_name is None:
        return None
    cols = marker_dict.get(marker_name)
    if cols is None or len(cols) <= ap_axis_idx:
        return None
    col = cols[ap_axis_idx]
    if col not in df.columns:
        return None
    return df[col].values.astype(float)


def _detect_ap_axis(df, marker_dict: dict, ref_marker: str) -> int:
    """Auto-detect the anterior-posterior gait axis (0=X, 1=Y, 2=Z) as the
    axis with the greatest range of motion for the reference marker."""
    cols = marker_dict.get(ref_marker)
    if cols is None:
        return 2  # fallback: Z
    max_span = -1
    best_idx = 2
    for idx, col in enumerate(cols[:3]):
        if col in df.columns:
            span = float(df[col].max() - df[col].min())
            if span > max_span:
                max_span = span
                best_idx = idx
    return best_idx


def _butter_lp(signal: np.ndarray, fs: float, cutoff: float = 6.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    if len(signal) <= 3 * order:
        return signal
    nyq = 0.5 * fs
    normal_cutoff = min(0.99, cutoff / nyq)
    b, a = butter(order, normal_cutoff, btype='low')
    padlen = min(15, len(signal) - 1)
    return filtfilt(b, a, signal, padlen=padlen)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_events_from_markers(trc) -> dict:
    """Detect heel strikes and toe-offs from TRC marker trajectories.

    Uses the Zeni et al. (2008) kinematic algorithm:
      - Heel Strike = local maximum of (heel_AP - pelvis_AP)
      - Toe-Off     = local minimum  of (toe_AP  - pelvis_AP)

    Works on a treadmill because the positions are expressed *relative* to
    the pelvis centre (which moves with the subject), removing the effect
    of the belt moving backward under the feet.

    Args:
        trc: TRC object with marker_dict and data attributes.

    Returns:
        dict with structure:
            {
              'HS': {'R': [t_seconds, ...], 'L': [t_seconds, ...]},
              'TO': {'R': [t_seconds, ...], 'L': [t_seconds, ...]},
            }
        Missing or undetected sides return empty lists.
    """
    result = {'HS': {'R': [], 'L': []}, 'TO': {'R': [], 'L': []}}

    if trc is None:
        print("[Hybrid] No TRC object available -- skipping marker-based detection.")
        return result

    df = trc.data
    marker_dict = trc.marker_dict
    t_trc = df['Time'].values.astype(float)

    if len(t_trc) < 10:
        print("[Hybrid] TRC data too short -- skipping marker-based detection.")
        return result

    fs = 1.0 / float(np.mean(np.diff(t_trc)))

    # ---- Find required markers ------------------------------------------------
    r_heel_name = _find_marker(marker_dict, _R_HEEL_CANDIDATES)
    l_heel_name = _find_marker(marker_dict, _L_HEEL_CANDIDATES)
    r_toe_name  = _find_marker(marker_dict, _R_TOE_CANDIDATES)
    l_toe_name  = _find_marker(marker_dict, _L_TOE_CANDIDATES)
    r_asi_name  = _find_marker(marker_dict, _R_ASI_CANDIDATES)
    l_asi_name  = _find_marker(marker_dict, _L_ASI_CANDIDATES)

    missing = [label for label, val in [('R heel', r_heel_name), ('L heel', l_heel_name),
                                         ('R toe',  r_toe_name),  ('L toe',  l_toe_name),
                                         ('RASI',   r_asi_name),  ('LASI',   l_asi_name)]
               if val is None]
    if missing:
        print(f"[Hybrid] Marker-based detection: missing {missing} -- "
              f"affected sides will use GRF-only events as suggested.")

    # ---- Auto-detect AP axis --------------------------------------------------
    ref_for_axis = r_heel_name or l_heel_name
    if ref_for_axis is None:
        print("[Hybrid] Cannot auto-detect AP axis (no heel marker found).")
        return result
    ap_idx = _detect_ap_axis(df, marker_dict, ref_for_axis)
    print(f"[Hybrid] AP axis: index {ap_idx} ({['X','Y','Z'][ap_idx]}) "
          f"from marker '{ref_for_axis}'")

    # ---- Pelvis AP centre (mean of RASI + LASI) -------------------------------
    r_asi_ap = _get_marker_ap(df, marker_dict, r_asi_name, ap_idx)
    l_asi_ap = _get_marker_ap(df, marker_dict, l_asi_name, ap_idx)

    if r_asi_ap is not None and l_asi_ap is not None:
        pelvis_ap = _butter_lp(0.5 * (r_asi_ap + l_asi_ap), fs)
    elif r_asi_ap is not None:
        pelvis_ap = _butter_lp(r_asi_ap, fs)
    elif l_asi_ap is not None:
        pelvis_ap = _butter_lp(l_asi_ap, fs)
    else:
        print("[Hybrid] RASI and LASI both missing -- cannot compute pelvis centre.")
        return result

    # Peak-finding parameters (based on test_zeni_algorithm.py)
    prominence  = 20
    min_dist_fr = int(0.8 * fs)   # at least 0.8 s between consecutive events

    # ---- RIGHT side -----------------------------------------------------------
    r_heel_ap = _get_marker_ap(df, marker_dict, r_heel_name, ap_idx)
    r_toe_ap  = _get_marker_ap(df, marker_dict, r_toe_name,  ap_idx)

    if r_heel_ap is not None:
        rel = _butter_lp(r_heel_ap - pelvis_ap, fs)
        idx, _ = find_peaks(rel, distance=min_dist_fr, prominence=prominence)
        result['HS']['R'] = [float(t_trc[i]) for i in idx if i < len(t_trc)]
        print(f"[Hybrid] Right HS (marker): {len(result['HS']['R'])} events")
    else:
        print("[Hybrid] Right heel marker not found -- Right HS skipped.")

    if r_toe_ap is not None:
        rel = _butter_lp(r_toe_ap - pelvis_ap, fs)
        idx, _ = find_peaks(-rel, distance=min_dist_fr, prominence=prominence)
        result['TO']['R'] = [float(t_trc[i]) for i in idx if i < len(t_trc)]
        print(f"[Hybrid] Right TO (marker): {len(result['TO']['R'])} events")
    else:
        print("[Hybrid] Right toe marker not found -- Right TO skipped.")

    # ---- LEFT side ------------------------------------------------------------
    l_heel_ap = _get_marker_ap(df, marker_dict, l_heel_name, ap_idx)
    l_toe_ap  = _get_marker_ap(df, marker_dict, l_toe_name,  ap_idx)

    if l_heel_ap is not None:
        rel = _butter_lp(l_heel_ap - pelvis_ap, fs)
        idx, _ = find_peaks(rel, distance=min_dist_fr, prominence=prominence)
        result['HS']['L'] = [float(t_trc[i]) for i in idx if i < len(t_trc)]
        print(f"[Hybrid] Left  HS (marker): {len(result['HS']['L'])} events")
    else:
        print("[Hybrid] Left heel marker not found -- Left HS skipped.")

    if l_toe_ap is not None:
        rel = _butter_lp(l_toe_ap - pelvis_ap, fs)
        idx, _ = find_peaks(-rel, distance=min_dist_fr, prominence=prominence)
        result['TO']['L'] = [float(t_trc[i]) for i in idx if i < len(t_trc)]
        print(f"[Hybrid] Left  TO (marker): {len(result['TO']['L'])} events")
    else:
        print("[Hybrid] Left toe marker not found -- Left TO skipped.")

    return result


def reconcile_events(grf_times: list,
                     marker_times: list,
                     tolerance_s: float = 0.1) -> tuple:
    """Reconcile GRF-detected and marker-detected events for one side + type.

    Matching logic:
      - GRF event + marker event within +-tolerance_s  -> CONFIRMED (GRF time)
      - GRF event with no marker match                 -> SUGGESTED (GRF-only)
      - Marker event with no GRF match                 -> SUGGESTED (marker-only)

    Args:
        grf_times:    Timestamps from GRF detection (seconds).
        marker_times: Timestamps from Zeni marker detection (seconds).
        tolerance_s:  Match tolerance in seconds (default 0.1 = 100 ms).

    Returns:
        (confirmed, suggested) -- two sorted lists of float timestamps.
        confirmed: use these for segmentation automatically.
        suggested: show as ghost markers in the GUI; ignored unless promoted.
    """
    grf_arr    = np.array(sorted(grf_times),    dtype=float)
    marker_arr = np.array(sorted(marker_times), dtype=float)

    confirmed: list = []
    suggested: list = []
    marker_matched = np.zeros(len(marker_arr), dtype=bool)

    for grf_t in grf_arr:
        if len(marker_arr) == 0:
            suggested.append(float(grf_t))
            continue
        diffs = np.abs(marker_arr - grf_t)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= tolerance_s and not marker_matched[best_idx]:
            confirmed.append(float(grf_t))     # both agree -> keep GRF timestamp
            marker_matched[best_idx] = True
        else:
            suggested.append(float(grf_t))     # GRF-only

    # Unmatched marker-only events -> suggested
    for idx, m_t in enumerate(marker_arr):
        if not marker_matched[idx]:
            suggested.append(float(m_t))

    confirmed.sort()
    suggested.sort()
    return confirmed, suggested
