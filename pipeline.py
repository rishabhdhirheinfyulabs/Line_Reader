"""Signal processing pipeline for strip analysis."""

import numpy as np
from pybaselines.whittaker import asls
from scipy.signal import find_peaks, savgol_filter

from utils import classify_ratio, get_regions, interleaved_average_signal


ALS_LAMBDA = 1e7
ALS_P = 0.999
SMOOTH_WINDOW = 15
T1_INNER_TRIM_FRACTION = 0.25
WIDTH_DIP_EPS = 0.3
WIDTH_FLAT_STEPS = 3


def _effective_sg_window(n_points, requested):
    if n_points < 3:
        return None
    w = min(int(requested), int(n_points))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return None
    return w


def correct_baseline(signal, lam=ALS_LAMBDA, p=ALS_P, smooth_win=SMOOTH_WINDOW):
    """SG smoothing + ALS baseline + inversion."""
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    sg_w = _effective_sg_window(n, smooth_win)
    if sg_w is None:
        smoothed = signal.copy()
    else:
        smoothed = savgol_filter(signal, window_length=sg_w, polyorder=2)

    trim_n = int(sg_w) if sg_w is not None else 0
    max_trim_each_side = max(0, (n - 5) // 2)
    trim_n = min(trim_n, max_trim_each_side)
    start_idx = trim_n
    end_idx = n - trim_n
    if end_idx - start_idx < 5:
        start_idx = 0
        end_idx = n

    fit_idx = np.arange(start_idx, end_idx, dtype=int)
    smoothed_fit = smoothed[start_idx:end_idx]
    baseline_fit, _ = asls(smoothed_fit, lam=lam, p=p)
    baseline = np.interp(np.arange(n), fit_idx, baseline_fit)
    corrected = baseline - smoothed
    return smoothed, baseline, corrected


def _inner_slice(indices, trim_fraction):
    n = len(indices)
    if n == 0:
        return indices
    trim_n = int(np.floor(float(trim_fraction) * n))
    if n - 2 * trim_n < 1:
        return indices
    return indices[trim_n:n - trim_n]


def find_dip(y, peak_idx, direction):
    """Slope-based dip detection from selected peak."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3 or peak_idx <= 0 or peak_idx >= n - 1:
        return None

    eps = float(WIDTH_DIP_EPS)
    flat_need = int(max(1, WIDTH_FLAT_STEPS))
    had_decrease = False
    flat_run = 0
    first_flat_idx = None

    if direction == "right":
        for i in range(peak_idx + 1, n):
            diff = float(y[i] - y[i - 1])
            if diff < -eps:
                had_decrease = True
                flat_run = 0
                first_flat_idx = None
            elif abs(diff) < eps:
                if flat_run == 0:
                    first_flat_idx = i - 1
                flat_run += 1
                if flat_run >= flat_need:
                    return int(first_flat_idx)
            elif diff > eps:
                if had_decrease:
                    return int(i - 1)
                flat_run = 0
                first_flat_idx = None
            else:
                flat_run = 0
                first_flat_idx = None
        return None

    if direction == "left":
        for i in range(peak_idx - 1, -1, -1):
            diff = float(y[i] - y[i + 1])
            if diff < -eps:
                had_decrease = True
                flat_run = 0
                first_flat_idx = None
            elif abs(diff) < eps:
                if flat_run == 0:
                    first_flat_idx = i + 1
                flat_run += 1
                if flat_run >= flat_need:
                    return int(first_flat_idx)
            elif diff > eps:
                if had_decrease:
                    return int(i + 1)
                flat_run = 0
                first_flat_idx = None
            else:
                flat_run = 0
                first_flat_idx = None
        return None

    return None


def _calc_width_from_dips(t, corrected, region_idx, peak_global_idx):
    if region_idx.size < 3:
        return np.nan, np.nan, np.nan, np.nan
    local_matches = np.where(region_idx == int(peak_global_idx))[0]
    if local_matches.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    y_region = np.asarray(corrected[region_idx], dtype=float)
    local_peak = int(local_matches[0])
    left_local = find_dip(y_region, local_peak, "left")
    right_local = find_dip(y_region, local_peak, "right")
    if left_local is None or right_local is None:
        return np.nan, np.nan, np.nan, np.nan

    left_global = int(region_idx[left_local])
    right_global = int(region_idx[right_local])
    if right_global <= left_global:
        return np.nan, np.nan, np.nan, np.nan

    width = float(t[right_global] - t[left_global])
    width_y = float(max(corrected[left_global], corrected[right_global]))
    return left_global, right_global, width, width_y


def _find_peak_in_region(t, corrected, line_label, region_start, region_end):
    mask = np.isfinite(t) & np.isfinite(corrected) & (t >= region_start) & (t <= region_end)
    region_idx = np.where(mask)[0]
    if region_idx.size < 3:
        return {
            "line_label": line_label,
            "peak_found": False,
            "peak_index": np.nan,
            "peak_time": np.nan,
            "peak_height": np.nan,
            "peak_prominence": np.nan,
            "width_found": False,
            "width_left_index": np.nan,
            "width_right_index": np.nan,
            "width_left_time": np.nan,
            "width_right_time": np.nan,
            "peak_width": np.nan,
            "width_y": np.nan,
        }

    y = np.asarray(corrected, dtype=float)
    if line_label == "T1":
        core_idx = _inner_slice(region_idx, T1_INNER_TRIM_FRACTION)
        if core_idx.size == 0:
            core_idx = region_idx
        if not np.isfinite(y[core_idx]).any():
            return {
                "line_label": line_label,
                "peak_found": False,
                "peak_index": np.nan,
                "peak_time": np.nan,
                "peak_height": np.nan,
                "peak_prominence": np.nan,
                "width_found": False,
                "width_left_index": np.nan,
                "width_right_index": np.nan,
                "width_left_time": np.nan,
                "width_right_time": np.nan,
                "peak_width": np.nan,
                "width_y": np.nan,
            }
        best_global = int(core_idx[np.nanargmax(y[core_idx])])
        prom = np.nan
    else:
        peaks, props = find_peaks(y[region_idx], prominence=1.0)
        if peaks.size == 0:
            best_global = int(region_idx[np.nanargmax(y[region_idx])])
            prom = np.nan
        else:
            p = props.get("prominences", np.zeros_like(peaks, dtype=float))
            best_local = int(peaks[np.argmax(p)])
            best_global = int(region_idx[best_local])
            prom = float(np.nanmax(p))

    li, ri, w, wy = _calc_width_from_dips(t, y, region_idx, best_global)
    return {
        "line_label": line_label,
        "peak_found": True,
        "peak_index": best_global,
        "peak_time": float(t[best_global]),
        "peak_height": float(y[best_global]),
        "peak_prominence": prom,
        "width_found": bool(np.isfinite(w)),
        "width_left_index": li,
        "width_right_index": ri,
        "width_left_time": float(t[int(li)]) if np.isfinite(li) else np.nan,
        "width_right_time": float(t[int(ri)]) if np.isfinite(ri) else np.nan,
        "peak_width": w,
        "width_y": wy,
    }


def run_pipeline(t_raw, y_raw, stride, direction, region_cfg, mode, smooth_window=SMOOTH_WINDOW):
    """Execute full pipeline and return structured results."""
    t_raw = np.asarray(t_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    if int(stride) == 0:
        t_proc, y_proc, parts = t_raw, y_raw, []
    else:
        t_proc, y_proc, parts = interleaved_average_signal(t_raw, y_raw, int(stride))

    smoothed, baseline, corrected = correct_baseline(y_proc, smooth_win=smooth_window)

    labels = get_regions(mode, direction)
    regions = []
    peaks = []
    for label in labels:
        center = float(region_cfg[label]["center"])
        width = float(region_cfg[label]["width"])
        start = center - width / 2.0
        end = center + width / 2.0
        regions.append((label, start, end))
        peaks.append(_find_peak_in_region(t_proc, corrected, label, start, end))

    peaks_by_label = {p["line_label"]: p for p in peaks}
    cl = peaks_by_label.get("CL", {})
    t1 = peaks_by_label.get("T1", {})

    cl_height = cl.get("peak_height", np.nan)
    t1_height = t1.get("peak_height", np.nan)
    cl_width = cl.get("peak_width", np.nan)
    t1_width = t1.get("peak_width", np.nan)
    cl_wy = cl.get("width_y", np.nan)
    t1_wy = t1.get("width_y", np.nan)

    # T missing -> T=0 for classification ratios.
    t1_height_safe = 0.0 if not np.isfinite(t1_height) else float(t1_height)
    t1_h_wb = 0.0 if (not np.isfinite(t1_height) or not np.isfinite(t1_wy)) else float(t1_height - t1_wy)
    cl_h_wb = (cl_height - cl_wy) if (np.isfinite(cl_height) and np.isfinite(cl_wy)) else np.nan

    tc_u = (t1_height_safe / cl_height) if (np.isfinite(cl_height) and cl_height != 0) else np.nan
    tc_w = (t1_h_wb / cl_h_wb) if (np.isfinite(cl_h_wb) and cl_h_wb != 0) else np.nan
    cl_present = bool(np.isfinite(cl_height) and cl_height > 0)

    return {
        "times": t_proc,
        "raw": y_proc,
        "split_series": parts,
        "smoothed": smoothed,
        "baseline": baseline,
        "corrected": corrected,
        "regions": regions,
        "region_labels": labels,
        "peaks": peaks,
        "peaks_by_label": peaks_by_label,
        "T1_peak_time": t1.get("peak_time", np.nan),
        "T1_peak_height": t1_height,
        "CL_peak_time": cl.get("peak_time", np.nan),
        "CL_peak_height": cl_height,
        "T1_peak_width": t1_width,
        "CL_peak_width": cl_width,
        "T1_width_baseline_y": t1_wy,
        "CL_width_baseline_y": cl_wy,
        "T1_height_width_baseline": t1_h_wb,
        "CL_height_width_baseline": cl_h_wb,
        "T_C_ratio_universal_baseline": tc_u,
        "T_C_ratio_width_baseline": tc_w,
        "strip_class_universal": classify_ratio(tc_u, cl_present),
        "strip_class_width": classify_ratio(tc_w, cl_present),
        "is_valid_test": cl_present,
    }

