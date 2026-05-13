"""Utility helpers for Streamlit strip analysis app."""

from io import BytesIO
import zipfile

import numpy as np
import pandas as pd


MODE_OPTIONS = ["AFM", "3 in one", "4 in one"]
STRIDE_OPTIONS = list(range(0, 11))
DEFAULT_STRIDE = 4


def get_regions(mode, direction):
    """Return ordered region labels for the selected mode/direction."""
    direction = str(direction).strip().lower()
    if mode == "AFM":
        return ["CL", "T1"] if direction == "forward" else ["T1", "CL"]
    if mode == "3 in one":
        return ["CL", "T3", "T2", "T1"] if direction == "forward" else ["T1", "T2", "T3", "CL"]
    if mode == "4 in one":
        return ["CL", "T4", "T3", "T2", "T1"] if direction == "forward" else ["T1", "T2", "T3", "T4", "CL"]
    # Safe fallback.
    return ["CL", "T1"] if direction == "forward" else ["T1", "CL"]


def load_data(uploaded_file):
    """Load CSV and return cleaned t plus selectable y columns."""
    df = pd.read_csv(uploaded_file)
    df.columns = [str(c).strip() for c in df.columns]

    if "t" not in df.columns:
        raise ValueError("CSV must include a 't' column.")

    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    t_mask = np.isfinite(t)
    t = t[t_mask]
    if t.size < 5:
        raise ValueError("Too few valid x-axis points after removing NaNs.")

    # All numeric columns except t are valid run choices.
    y_columns = [
        c for c in df.columns
        if c != "t" and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    if not y_columns:
        raise ValueError("No valid Y-axis columns found (numeric columns besides 't').")

    # Default: immediate next column to t if possible.
    t_idx = list(df.columns).index("t")
    default_y_col = None
    if t_idx + 1 < len(df.columns):
        cand = df.columns[t_idx + 1]
        if cand in y_columns:
            default_y_col = cand
    if default_y_col is None:
        default_y_col = y_columns[0]

    return {"df": df, "t": t, "y_columns": y_columns, "default_y_col": default_y_col}


def interleaved_average_signal(t, y, n_groups):
    """Split into n interleaved sub-signals and average to one."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n_groups = int(max(1, n_groups))

    parts = []
    for g in range(n_groups):
        tg = t[g::n_groups]
        yg = y[g::n_groups]
        m = np.isfinite(tg) & np.isfinite(yg)
        tg = tg[m]
        yg = yg[m]
        if tg.size > 0:
            parts.append((tg, yg))

    if not parts:
        return t, y, []

    min_len = min(len(v) for _, v in parts)
    if min_len < 3:
        return t, y, []

    t_stack = np.vstack([tt[:min_len] for tt, _ in parts])
    y_stack = np.vstack([yy[:min_len] for _, yy in parts])
    t_avg = np.nanmean(t_stack, axis=0)
    y_avg = np.nanmean(y_stack, axis=0)
    trimmed_parts = [(tt[:min_len], yy[:min_len]) for tt, yy in parts]
    return t_avg, y_avg, trimmed_parts


def classify_ratio(tc_ratio, cl_present):
    """Classification rules for universal/width baseline ratios."""
    if not cl_present or not np.isfinite(tc_ratio):
        return "INVALID"
    if tc_ratio > 1.1:
        return "NEGATIVE"
    if 0.9 <= tc_ratio <= 1.1:
        return "WEAK POSITIVE"
    return "POSITIVE"


def init_region_defaults(t, labels):
    """Initialize region centers/widths, with AFM-specific defaults."""
    t = np.asarray(t, dtype=float)
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    span = max(1e-6, tmax - tmin)

    # AFM defaults requested by user:
    # CL center=1.90, T1 center=3.50, shared width=0.75.
    # Keep label-order compatibility for forward/backward layouts.
    labels_upper = [str(label).upper() for label in labels]
    if set(labels_upper) == {"CL", "T1"} and len(labels_upper) == 2:
        afm_centers = {"CL": 1.90, "T1": 3.50}
        afm_width = 0.75
        return {
            label: {
                "center": float(afm_centers[str(label).upper()]),
                "width": float(afm_width),
            }
            for label in labels
        }

    n = max(1, len(labels))
    centers = np.linspace(tmin + 0.2 * span, tmin + 0.8 * span, n)
    width = 0.18 * span
    return {
        label: {"center": float(c), "width": float(width)}
        for label, c in zip(labels, centers)
    }


def flatten_result(result):
    """Flatten nested result dict for debug/show-more tables."""
    out = {}
    for k, v in result.items():
        if isinstance(v, (list, tuple, np.ndarray, dict)):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def build_results_zip(minimal_row, full_row, figs_dict, extra_tables=None):
    """Build zip bytes containing reports and PNG plots."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report_minimal.csv", pd.DataFrame([minimal_row]).to_csv(index=False))
        zf.writestr("report_full.csv", pd.DataFrame([full_row]).to_csv(index=False))
        if extra_tables:
            for name, table in extra_tables.items():
                if table is None:
                    continue
                if isinstance(table, pd.DataFrame):
                    zf.writestr(name, table.to_csv(index=False))
                else:
                    zf.writestr(name, pd.DataFrame(table).to_csv(index=False))
        for name, fig in figs_dict.items():
            fbuf = BytesIO()
            fig.savefig(fbuf, format="png", dpi=160, bbox_inches="tight")
            fbuf.seek(0)
            zf.writestr(name, fbuf.getvalue())
    buf.seek(0)
    return buf

