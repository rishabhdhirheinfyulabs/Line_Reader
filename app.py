"""Entry point for Streamlit app."""

from app_ui import main


if __name__ == "__main__":
    main()
    # Hard-stop so any accidental legacy content below never executes.
    import streamlit as st
    st.stop()

"""Entry point for Streamlit app."""

import streamlit as st

from app_ui import main


# Run modular UI and stop immediately to avoid executing any legacy content
# that might still exist below in this file.
main()
st.stop()

"""Streamlit UI for strip analysis (UI orchestration only)."""

import numpy as np
import pandas as pd
import streamlit as st

from pipeline import run_pipeline
from plotting import make_pipeline_plots, plot_live_signal
from utils import (
    DEFAULT_STRIDE,
    MODE_OPTIONS,
    STRIDE_OPTIONS,
    build_results_zip,
    flatten_result,
    get_regions,
    init_region_defaults,
    interleaved_average_signal,
    load_data,
)


DEFAULT_SG_SMOOTHING = 15


def _ensure_session_defaults():
    if "mode" not in st.session_state:
        st.session_state.mode = MODE_OPTIONS[0]
    if "stride" not in st.session_state:
        st.session_state.stride = DEFAULT_STRIDE
    if "direction" not in st.session_state:
        st.session_state.direction = "forward"
    if "smooth_window" not in st.session_state:
        st.session_state.smooth_window = DEFAULT_SG_SMOOTHING
    if "region_cfg" not in st.session_state:
        # region_cfg[mode][direction][label] = {center, width}
        st.session_state.region_cfg = {}
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "selected_y_col" not in st.session_state:
        st.session_state.selected_y_col = None


def _region_ui_color(label):
    return "#CFE8FF" if str(label).upper() == "CL" else "#FFF3C9"


def _ensure_mode_direction_regions(t, mode, direction):
    mode_cfg = st.session_state.region_cfg.setdefault(mode, {})
    dir_cfg = mode_cfg.setdefault(direction, {})
    labels = get_regions(mode, direction)
    missing = [lbl for lbl in labels if lbl not in dir_cfg]
    if missing:
        defaults = init_region_defaults(t, labels)
        for lbl in labels:
            if lbl not in dir_cfg:
                dir_cfg[lbl] = defaults[lbl]
    return dir_cfg


def _values_close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def _render_region_controls(mode, direction, t):
    """
    For each region:
    - center slider + number input (independent)
    - width slider + number input (all regions linked in same strip mode/direction)
    """
    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    span = max(1e-6, tmax - tmin)
    step = float(span / 500.0)

    cfg = _ensure_mode_direction_regions(t, mode, direction)
    labels = get_regions(mode, direction)
    changed_width_values = []

    for lbl in labels:
        bg = _region_ui_color(lbl)
        st.markdown(
            f"<div style='background:{bg};padding:6px 8px;border-radius:6px;'><b>{lbl} region</b></div>",
            unsafe_allow_html=True,
        )
        prev_center = float(cfg[lbl]["center"])
        prev_width = float(cfg[lbl]["width"])

        base = f"{mode}|{direction}|{lbl}"
        key_cs = f"{base}|center|state|slider"
        key_cn = f"{base}|center|state|number"
        key_ws = f"{base}|width|state|slider"
        key_wn = f"{base}|width|state|number"

        if key_cs not in st.session_state:
            st.session_state[key_cs] = prev_center
        if key_cn not in st.session_state:
            st.session_state[key_cn] = prev_center
        if key_ws not in st.session_state:
            st.session_state[key_ws] = prev_width
        if key_wn not in st.session_state:
            st.session_state[key_wn] = prev_width

        c1, c2 = st.columns(2)
        with c1:
            s_center = st.slider(
                f"{lbl} center (slider)",
                min_value=tmin,
                max_value=tmax,
                value=float(st.session_state[key_cs]),
                step=step,
                key=f"{base}|center|widget|slider",
            )
        with c2:
            n_center = st.number_input(
                f"{lbl} center (number)",
                min_value=tmin,
                max_value=tmax,
                value=float(st.session_state[key_cn]),
                step=step,
                key=f"{base}|center|widget|number",
            )

        c3, c4 = st.columns(2)
        with c3:
            s_width = st.slider(
                f"{lbl} width (slider)",
                min_value=float(span / 200.0),
                max_value=float(span),
                value=float(st.session_state[key_ws]),
                step=step,
                key=f"{base}|width|widget|slider",
            )
        with c4:
            n_width = st.number_input(
                f"{lbl} width (number)",
                min_value=float(span / 200.0),
                max_value=float(span),
                value=float(st.session_state[key_wn]),
                step=step,
                key=f"{base}|width|widget|number",
            )

        # Sync center slider/number (independent per region)
        if not _values_close(s_center, prev_center) and _values_close(n_center, prev_center):
            new_center = float(s_center)
        elif not _values_close(n_center, prev_center) and _values_close(s_center, prev_center):
            new_center = float(n_center)
        elif not _values_close(s_center, prev_center) and not _values_close(n_center, prev_center):
            new_center = float(n_center)
        else:
            new_center = prev_center

        # Detect width edits for global coupling
        if not _values_close(s_width, prev_width) and _values_close(n_width, prev_width):
            new_width = float(s_width)
            changed_width_values.append(new_width)
        elif not _values_close(n_width, prev_width) and _values_close(s_width, prev_width):
            new_width = float(n_width)
            changed_width_values.append(new_width)
        elif not _values_close(s_width, prev_width) and not _values_close(n_width, prev_width):
            new_width = float(n_width)
            changed_width_values.append(new_width)
        else:
            new_width = prev_width

        cfg[lbl]["center"] = float(new_center)
        cfg[lbl]["width"] = float(new_width)

        # Keep state mirrors in sync
        st.session_state[key_cs] = new_center
        st.session_state[key_cn] = new_center
        st.session_state[key_ws] = new_width
        st.session_state[key_wn] = new_width

    # Width coupling across all regions in this strip mode+direction.
    if labels:
        if changed_width_values:
            coupled = float(changed_width_values[-1])  # latest edited value wins
        else:
            coupled = float(cfg[labels[0]]["width"])

        for lbl in labels:
            cfg[lbl]["width"] = coupled
            base = f"{mode}|{direction}|{lbl}"
            st.session_state[f"{base}|width|state|slider"] = coupled
            st.session_state[f"{base}|width|state|number"] = coupled

    return cfg


def _select_y_column(df, y_columns, default_y_col):
    # Keep existing selection if still valid.
    if st.session_state.selected_y_col not in y_columns:
        st.session_state.selected_y_col = default_y_col

    selected = st.selectbox(
        "Y-axis run column",
        y_columns,
        index=y_columns.index(st.session_state.selected_y_col),
        key="y_column_selectbox",
    )
    st.session_state.selected_y_col = selected
    y = pd.to_numeric(df[selected], errors="coerce").to_numpy(dtype=float)
    return selected, y


def main():
    st.set_page_config(page_title="LFA Strip Analyzer", layout="wide")
    st.title("Lateral Flow Strip Signal Analyzer")
    _ensure_session_defaults()

    col_left, col_right = st.columns([1, 4], gap="large")

    with col_left:
        with st.expander("Controls", expanded=True):
            st.session_state.mode = st.selectbox(
                "Mode",
                MODE_OPTIONS,
                index=MODE_OPTIONS.index(st.session_state.mode),
                key="mode_selectbox",
            )
            uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
            st.session_state.stride = st.selectbox(
                "Stride",
                STRIDE_OPTIONS,
                index=STRIDE_OPTIONS.index(st.session_state.stride),
                key="stride_selectbox",
            )
            st.session_state.smooth_window = int(
                st.number_input(
                    "SG smoothing window",
                    min_value=3,
                    max_value=301,
                    value=int(st.session_state.smooth_window),
                    step=2,
                    key="sg_smoothing_number",
                )
            )
            st.session_state.direction = st.radio(
                "Direction",
                ["forward", "backward"],
                horizontal=True,
                index=0 if st.session_state.direction == "forward" else 1,
                key="direction_radio",
            )
            analyze_clicked = st.button("Analyze", use_container_width=True, key="analyze_button")
            show_more = st.toggle("Show more", value=False, key="show_more_toggle")

    if uploaded is None:
        with col_right:
            st.info("Upload a CSV with columns like `t, run1` to start.")
        return

    try:
        data = load_data(uploaded)
    except Exception as exc:
        with col_right:
            st.error(f"Failed to load file: {exc}")
        return

    df = data["df"]
    t_raw = data["t"]
    y_columns = data["y_columns"]
    default_y_col = data["default_y_col"]
    mode = st.session_state.mode
    direction = st.session_state.direction

    with col_left:
        selected_y_col, y_raw = _select_y_column(df, y_columns, default_y_col)

    # Align selected y with cleaned t (same row-wise finite mask on t and y).
    t_full = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t_full) & np.isfinite(y_raw)
    t_raw = t_full[mask]
    y_raw = y_raw[mask]
    if t_raw.size < 5:
        with col_right:
            st.error("Too few valid points after applying selected Y column.")
        return

    with col_left:
        with st.expander(f"Region Controls ({mode} | {direction})", expanded=True):
            region_cfg = _render_region_controls(mode, direction, t_raw)

    # Live updates on upload / run-column / stride / region changes.
    if int(st.session_state.stride) == 0:
        t_live, y_live, split_live = t_raw, y_raw, []
    else:
        t_live, y_live, split_live = interleaved_average_signal(t_raw, y_raw, st.session_state.stride)

    live_regions = []
    for lbl in get_regions(mode, direction):
        c = float(region_cfg[lbl]["center"])
        w = float(region_cfg[lbl]["width"])
        live_regions.append((lbl, c - w / 2.0, c + w / 2.0))

    with col_right:
        st.subheader("Live Signal")
        fig_live = plot_live_signal(
            t_live,
            y_live,
            split_live,
            live_regions,
            title=(
                f"Mode={mode} | y={selected_y_col} | direction={direction} | "
                f"stride={st.session_state.stride} | SG={st.session_state.smooth_window}"
            ),
        )
        st.pyplot(fig_live, use_container_width=True)

    if analyze_clicked:
        st.session_state.analysis_result = run_pipeline(
            t_raw=t_raw,
            y_raw=y_raw,
            stride=st.session_state.stride,
            direction=direction,
            region_cfg=region_cfg,
            mode=mode,
            smooth_window=st.session_state.smooth_window,
        )

    res = st.session_state.analysis_result
    if res is None:
        return

    minimal_keys = [
        "T1_peak_time",
        "T1_peak_height",
        "CL_peak_time",
        "CL_peak_height",
        "T_C_ratio_universal_baseline",
        "T_C_ratio_width_baseline",
        "strip_class_universal",
        "strip_class_width",
    ]
    minimal_row = {k: res.get(k, np.nan) for k in minimal_keys}
    full_row = flatten_result(res)

    with col_right:
        st.subheader("Results")
        st.dataframe(pd.DataFrame([minimal_row]), use_container_width=True)
        if show_more:
            st.dataframe(pd.DataFrame([full_row]), use_container_width=True)

        st.subheader("Analysis Plots")
        title_prefix = (
            f"Mode={mode} | y={selected_y_col} | direction={direction} | stride={st.session_state.stride} | "
            f"SG={st.session_state.smooth_window} | "
            f"T/Cu={res['T_C_ratio_universal_baseline']:.3f} | T/Cw={res['T_C_ratio_width_baseline']:.3f} |"
        )
        fig1, fig2, fig3 = make_pipeline_plots(res, title_prefix=title_prefix)
        st.pyplot(fig1, use_container_width=True)
        st.pyplot(fig2, use_container_width=True)
        st.pyplot(fig3, use_container_width=True)

        zip_bytes = build_results_zip(
            minimal_row=minimal_row,
            full_row=full_row,
            figs_dict={
                "plot_live.png": fig_live,
                "plot_1_signal_baseline.png": fig1,
                "plot_2_corrected_peaks.png": fig2,
                "plot_3_flat_baseline.png": fig3,
            },
        )
        st.download_button(
            "Download Results",
            data=zip_bytes,
            file_name="strip_analysis_results.zip",
            mime="application/zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

"""Streamlit UI for lateral flow strip signal analysis (UI-only orchestration)."""

import numpy as np
import pandas as pd
import streamlit as st

from pipeline import run_pipeline
from plotting import make_pipeline_plots, plot_live_signal
from utils import (
    DEFAULT_STRIDE,
    MODE_OPTIONS,
    STRIDE_OPTIONS,
    build_results_zip,
    flatten_result,
    get_regions,
    init_region_defaults,
    interleaved_average_signal,
    load_data,
)


def _ensure_session_defaults():
    if "mode" not in st.session_state:
        st.session_state.mode = MODE_OPTIONS[0]
    if "stride" not in st.session_state:
        st.session_state.stride = DEFAULT_STRIDE
    if "direction" not in st.session_state:
        st.session_state.direction = "forward"
    if "region_cfg" not in st.session_state:
        # region_cfg[mode][direction][label] = {center, width}
        st.session_state.region_cfg = {}
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None


def _ensure_mode_direction_regions(t, mode, direction):
    mode_cfg = st.session_state.region_cfg.setdefault(mode, {})
    dir_cfg = mode_cfg.setdefault(direction, {})
    labels = get_regions(mode, direction)
    missing = [lbl for lbl in labels if lbl not in dir_cfg]
    if missing:
        defaults = init_region_defaults(t, labels)
        for lbl in labels:
            if lbl not in dir_cfg:
                dir_cfg[lbl] = defaults[lbl]
    return dir_cfg


def _values_close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def _render_region_controls(mode, direction, t):
    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    span = max(1e-6, tmax - tmin)
    step = float(span / 500.0)

    cfg = _ensure_mode_direction_regions(t, mode, direction)
    labels = get_regions(mode, direction)

    # Snapshot previous widths for AFM coupling.
    prev_widths = {lbl: float(cfg[lbl]["width"]) for lbl in labels if lbl in cfg}
    changed_width = {lbl: False for lbl in labels}

    for lbl in labels:
        st.markdown(f"**{lbl} region**")
        prev_center = float(cfg[lbl]["center"])
        prev_width = float(cfg[lbl]["width"])

        base = f"{mode}|{direction}|{lbl}"
        key_cs = f"{base}|center|slider"
        key_cn = f"{base}|center|number"
        key_ws = f"{base}|width|slider"
        key_wn = f"{base}|width|number"

        if key_cs not in st.session_state:
            st.session_state[key_cs] = prev_center
        if key_cn not in st.session_state:
            st.session_state[key_cn] = prev_center
        if key_ws not in st.session_state:
            st.session_state[key_ws] = prev_width
        if key_wn not in st.session_state:
            st.session_state[key_wn] = prev_width

        c1, c2 = st.columns(2)
        with c1:
            s_center = st.slider(
                f"{lbl} center (slider)",
                min_value=tmin,
                max_value=tmax,
                value=float(st.session_state[key_cs]),
                step=step,
                key=f"{key_cs}|widget",
            )
        with c2:
            n_center = st.number_input(
                f"{lbl} center (number)",
                min_value=tmin,
                max_value=tmax,
                value=float(st.session_state[key_cn]),
                step=step,
                key=f"{key_cn}|widget",
            )

        c3, c4 = st.columns(2)
        with c3:
            s_width = st.slider(
                f"{lbl} width (slider)",
                min_value=float(span / 200.0),
                max_value=float(span),
                value=float(st.session_state[key_ws]),
                step=step,
                key=f"{key_ws}|widget",
            )
        with c4:
            n_width = st.number_input(
                f"{lbl} width (number)",
                min_value=float(span / 200.0),
                max_value=float(span),
                value=float(st.session_state[key_wn]),
                step=step,
                key=f"{key_wn}|widget",
            )

        # Sync center slider/number: whichever changed from previous wins.
        if not _values_close(s_center, prev_center) and _values_close(n_center, prev_center):
            new_center = float(s_center)
        elif not _values_close(n_center, prev_center) and _values_close(s_center, prev_center):
            new_center = float(n_center)
        elif not _values_close(s_center, prev_center) and not _values_close(n_center, prev_center):
            new_center = float(n_center)  # typed value wins if both moved
        else:
            new_center = prev_center

        # Sync width slider/number.
        if not _values_close(s_width, prev_width) and _values_close(n_width, prev_width):
            new_width = float(s_width)
            changed_width[lbl] = True
        elif not _values_close(n_width, prev_width) and _values_close(s_width, prev_width):
            new_width = float(n_width)
            changed_width[lbl] = True
        elif not _values_close(s_width, prev_width) and not _values_close(n_width, prev_width):
            new_width = float(n_width)
            changed_width[lbl] = True
        else:
            new_width = prev_width

        cfg[lbl]["center"] = float(new_center)
        cfg[lbl]["width"] = float(new_width)

        # Push synchronized values back into both controls for next rerun.
        st.session_state[key_cs] = new_center
        st.session_state[key_cn] = new_center
        st.session_state[key_ws] = new_width
        st.session_state[key_wn] = new_width

    # AFM-only width coupling between CL and T1.
    if mode == "AFM (default)" and "CL" in cfg and "T1" in cfg:
        cl_prev = prev_widths.get("CL", cfg["CL"]["width"])
        t1_prev = prev_widths.get("T1", cfg["T1"]["width"])
        cl_now = float(cfg["CL"]["width"])
        t1_now = float(cfg["T1"]["width"])

        if changed_width.get("CL", False) and not changed_width.get("T1", False):
            coupled = cl_now
        elif changed_width.get("T1", False) and not changed_width.get("CL", False):
            coupled = t1_now
        elif changed_width.get("CL", False) and changed_width.get("T1", False):
            coupled = t1_now  # latest typed wins in tie
        else:
            coupled = cl_now if not _values_close(cl_now, cl_prev) else t1_now if not _values_close(t1_now, t1_prev) else cl_now

        cfg["CL"]["width"] = float(coupled)
        cfg["T1"]["width"] = float(coupled)

        # Also sync widget backing state.
        for lbl in ("CL", "T1"):
            base = f"{mode}|{direction}|{lbl}"
            st.session_state[f"{base}|width|slider"] = float(coupled)
            st.session_state[f"{base}|width|number"] = float(coupled)

    return cfg


def main():
    st.set_page_config(page_title="LFA Strip Analyzer", layout="wide")
    st.title("Lateral Flow Strip Signal Analyzer")
    _ensure_session_defaults()

    col_left, col_right = st.columns([1, 4], gap="large")

    with col_left:
        with st.expander("Controls", expanded=True):
            st.session_state.mode = st.selectbox(
                "Mode",
                MODE_OPTIONS,
                index=MODE_OPTIONS.index(st.session_state.mode),
                key="mode_selectbox",
            )
            uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
            st.session_state.stride = st.selectbox(
                "Stride",
                STRIDE_OPTIONS,
                index=STRIDE_OPTIONS.index(st.session_state.stride),
                key="stride_selectbox",
            )
            st.session_state.direction = st.radio(
                "Direction",
                ["forward", "backward"],
                horizontal=True,
                index=0 if st.session_state.direction == "forward" else 1,
                key="direction_radio",
            )
            analyze_clicked = st.button("Analyze", use_container_width=True, key="analyze_button")
            show_more = st.toggle("Show more", value=False, key="show_more_toggle")

    if uploaded is None:
        with col_right:
            st.info("Upload a CSV with columns like `t, run1` to start.")
        return

    try:
        data = load_data(uploaded)
    except Exception as exc:
        with col_right:
            st.error(f"Failed to load file: {exc}")
        return

    t_raw, y_raw, run_col = data["t"], data["y"], data["run_col"]
    mode = st.session_state.mode
    direction = st.session_state.direction

    with col_left:
        with st.expander(f"Region Controls ({mode} | {direction})", expanded=True):
            region_cfg = _render_region_controls(mode, direction, t_raw)

    # Live plot updates instantly on upload/stride/region changes.
    if int(st.session_state.stride) == 0:
        t_live, y_live, split_live = t_raw, y_raw, []
    else:
        t_live, y_live, split_live = interleaved_average_signal(t_raw, y_raw, st.session_state.stride)

    live_regions = []
    for lbl in get_regions(mode, direction):
        c = float(region_cfg[lbl]["center"])
        w = float(region_cfg[lbl]["width"])
        live_regions.append((lbl, c - w / 2.0, c + w / 2.0))

    with col_right:
        st.subheader("Live Signal")
        fig_live = plot_live_signal(
            t_live,
            y_live,
            split_live,
            live_regions,
            title=f"Mode={mode} | run={run_col} | direction={direction} | stride={st.session_state.stride}",
        )
        st.pyplot(fig_live, use_container_width=True)

    if analyze_clicked:
        st.session_state.analysis_result = run_pipeline(
            t_raw=t_raw,
            y_raw=y_raw,
            stride=st.session_state.stride,
            direction=direction,
            region_cfg=region_cfg,
            mode=mode,
        )

    res = st.session_state.analysis_result
    if res is None:
        return

    minimal_keys = [
        "T1_peak_time",
        "T1_peak_height",
        "CL_peak_time",
        "CL_peak_height",
        "T_C_ratio_universal_baseline",
        "T_C_ratio_width_baseline",
        "strip_class_universal",
        "strip_class_width",
    ]
    minimal_row = {k: res.get(k, np.nan) for k in minimal_keys}
    full_row = flatten_result(res)

    with col_right:
        st.subheader("Results")
        st.dataframe(pd.DataFrame([minimal_row]), use_container_width=True)
        if show_more:
            st.dataframe(pd.DataFrame([full_row]), use_container_width=True)

        st.subheader("Analysis Plots")
        title_prefix = (
            f"Mode={mode} | direction={direction} | stride={st.session_state.stride} | "
            f"T/Cu={res['T_C_ratio_universal_baseline']:.3f} | T/Cw={res['T_C_ratio_width_baseline']:.3f} |"
        )
        fig1, fig2, fig3 = make_pipeline_plots(res, title_prefix=title_prefix)
        st.pyplot(fig1, use_container_width=True)
        st.pyplot(fig2, use_container_width=True)
        st.pyplot(fig3, use_container_width=True)

        zip_bytes = build_results_zip(
            minimal_row=minimal_row,
            full_row=full_row,
            figs_dict={
                "plot_live.png": fig_live,
                "plot_1_signal_baseline.png": fig1,
                "plot_2_corrected_peaks.png": fig2,
                "plot_3_flat_baseline.png": fig3,
            },
        )
        st.download_button(
            "Download Results",
            data=zip_bytes,
            file_name="strip_analysis_results.zip",
            mime="application/zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

"""Streamlit UI for lateral flow strip signal analysis (UI-only orchestration)."""

import numpy as np
import pandas as pd
import streamlit as st

from pipeline import run_pipeline
from plotting import make_pipeline_plots, plot_live_signal
from utils import (
    DEFAULT_STRIDE,
    MODE_OPTIONS,
    STRIDE_OPTIONS,
    build_results_zip,
    flatten_result,
    get_regions,
    init_region_defaults,
    interleaved_average_signal,
    load_data,
)


def _ensure_session_defaults():
    if "mode" not in st.session_state:
        st.session_state.mode = MODE_OPTIONS[0]
    if "stride" not in st.session_state:
        st.session_state.stride = DEFAULT_STRIDE
    if "direction" not in st.session_state:
        st.session_state.direction = "forward"
    if "region_cfg" not in st.session_state:
        # region_cfg[mode][direction][label] = {center, width}
        st.session_state.region_cfg = {}
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None


def _ensure_mode_direction_regions(t, mode, direction):
    mode_cfg = st.session_state.region_cfg.setdefault(mode, {})
    dir_cfg = mode_cfg.setdefault(direction, {})
    labels = get_regions(mode, direction)
    missing = [lbl for lbl in labels if lbl not in dir_cfg]
    if missing:
        defaults = init_region_defaults(t, labels)
        for lbl in labels:
            if lbl not in dir_cfg:
                dir_cfg[lbl] = defaults[lbl]
    return dir_cfg


def _values_close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def _render_region_controls(mode, direction, t):
    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    span = max(1e-6, tmax - tmin)
    step = float(span / 500.0)

    cfg = _ensure_mode_direction_regions(t, mode, direction)
    labels = get_regions(mode, direction)

    # Snapshot previous widths for AFM coupling.
    prev_widths = {lbl: float(cfg[lbl]["width"]) for lbl in labels if lbl in cfg}
    changed_width = {lbl: False for lbl in labels}

    for lbl in labels:
        st.markdown(f"**{lbl} region**")
        prev_center = float(cfg[lbl]["center"])
        prev_width = float(cfg[lbl]["width"])

        base = f"{mode}|{direction}|{lbl}"
        key_cs = f"{base}|center|slider"
        key_cn = f"{base}|center|number"
        key_ws = f"{base}|width|slider"
        key_wn = f"{base}|width|number"

        if key_cs not in st.session_state:
            st.session_state[key_cs] = prev_center
        if key_cn not in st.session_state:
            st.session_state[key_cn] = prev_center
        if key_ws not in st.session_state:
            st.session_state[key_ws] = prev_width
        if key_wn not in st.session_state:
            st.session_state[key_wn] = prev_width

        c1, c2 = st.columns(2)
        with c1:
            s_center = st.slider(
                f"{lbl} center (slider)",
                min_value=tmin,
                max_value=tmax,
                value=float(st.session_state[key_cs]),
                step=step,
                key=f"{key_cs}|widget",
            )
        with c2:
            n_center = st.number_input(
                f"{lbl} center (number)",
                min_value=tmin,
                max_value=tmax,
                value=float(st.session_state[key_cn]),
                step=step,
                key=f"{key_cn}|widget",
            )

        c3, c4 = st.columns(2)
        with c3:
            s_width = st.slider(
                f"{lbl} width (slider)",
                min_value=float(span / 200.0),
                max_value=float(span),
                value=float(st.session_state[key_ws]),
                step=step,
                key=f"{key_ws}|widget",
            )
        with c4:
            n_width = st.number_input(
                f"{lbl} width (number)",
                min_value=float(span / 200.0),
                max_value=float(span),
                value=float(st.session_state[key_wn]),
                step=step,
                key=f"{key_wn}|widget",
            )

        # Sync center slider/number: whichever changed from previous wins.
        if not _values_close(s_center, prev_center) and _values_close(n_center, prev_center):
            new_center = float(s_center)
        elif not _values_close(n_center, prev_center) and _values_close(s_center, prev_center):
            new_center = float(n_center)
        elif not _values_close(s_center, prev_center) and not _values_close(n_center, prev_center):
            new_center = float(n_center)  # typed value wins if both moved
        else:
            new_center = prev_center

        # Sync width slider/number.
        if not _values_close(s_width, prev_width) and _values_close(n_width, prev_width):
            new_width = float(s_width)
            changed_width[lbl] = True
        elif not _values_close(n_width, prev_width) and _values_close(s_width, prev_width):
            new_width = float(n_width)
            changed_width[lbl] = True
        elif not _values_close(s_width, prev_width) and not _values_close(n_width, prev_width):
            new_width = float(n_width)
            changed_width[lbl] = True
        else:
            new_width = prev_width

        cfg[lbl]["center"] = float(new_center)
        cfg[lbl]["width"] = float(new_width)

        # Push synchronized values back into both controls for next rerun.
        st.session_state[key_cs] = new_center
        st.session_state[key_cn] = new_center
        st.session_state[key_ws] = new_width
        st.session_state[key_wn] = new_width

    # AFM-only width coupling between CL and T1.
    if mode == "AFM (default)" and "CL" in cfg and "T1" in cfg:
        cl_prev = prev_widths.get("CL", cfg["CL"]["width"])
        t1_prev = prev_widths.get("T1", cfg["T1"]["width"])
        cl_now = float(cfg["CL"]["width"])
        t1_now = float(cfg["T1"]["width"])

        if changed_width.get("CL", False) and not changed_width.get("T1", False):
            coupled = cl_now
        elif changed_width.get("T1", False) and not changed_width.get("CL", False):
            coupled = t1_now
        elif changed_width.get("CL", False) and changed_width.get("T1", False):
            coupled = t1_now  # latest typed wins in tie
        else:
            coupled = cl_now if not _values_close(cl_now, cl_prev) else t1_now if not _values_close(t1_now, t1_prev) else cl_now

        cfg["CL"]["width"] = float(coupled)
        cfg["T1"]["width"] = float(coupled)

        # Also sync widget backing state.
        for lbl in ("CL", "T1"):
            base = f"{mode}|{direction}|{lbl}"
            st.session_state[f"{base}|width|slider"] = float(coupled)
            st.session_state[f"{base}|width|number"] = float(coupled)

    return cfg


def main():
    st.set_page_config(page_title="LFA Strip Analyzer", layout="wide")
    st.title("Lateral Flow Strip Signal Analyzer")
    _ensure_session_defaults()

    col_left, col_right = st.columns([1, 4], gap="large")

    with col_left:
        with st.expander("Controls", expanded=True):
            st.session_state.mode = st.selectbox("Mode", MODE_OPTIONS, index=MODE_OPTIONS.index(st.session_state.mode))
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            st.session_state.stride = st.selectbox("Stride", STRIDE_OPTIONS, index=STRIDE_OPTIONS.index(st.session_state.stride))
            st.session_state.direction = st.radio(
                "Direction",
                ["forward", "backward"],
                horizontal=True,
                index=0 if st.session_state.direction == "forward" else 1,
            )
            analyze_clicked = st.button("Analyze", use_container_width=True)
            show_more = st.toggle("Show more", value=False)

    if uploaded is None:
        with col_right:
            st.info("Upload a CSV with columns like `t, run1` to start.")
        return

    try:
        data = load_data(uploaded)
    except Exception as exc:
        with col_right:
            st.error(f"Failed to load file: {exc}")
        return

    t_raw, y_raw, run_col = data["t"], data["y"], data["run_col"]
    mode = st.session_state.mode
    direction = st.session_state.direction

    with col_left:
        with st.expander(f"Region Controls ({mode} | {direction})", expanded=True):
            region_cfg = _render_region_controls(mode, direction, t_raw)

    # Live plot updates instantly on upload/stride/region changes.
    if int(st.session_state.stride) == 0:
        t_live, y_live, split_live = t_raw, y_raw, []
    else:
        t_live, y_live, split_live = interleaved_average_signal(t_raw, y_raw, st.session_state.stride)

    live_regions = []
    for lbl in get_regions(mode, direction):
        c = float(region_cfg[lbl]["center"])
        w = float(region_cfg[lbl]["width"])
        live_regions.append((lbl, c - w / 2.0, c + w / 2.0))

    with col_right:
        st.subheader("Live Signal")
        fig_live = plot_live_signal(
            t_live,
            y_live,
            split_live,
            live_regions,
            title=f"Mode={mode} | run={run_col} | direction={direction} | stride={st.session_state.stride}",
        )
        st.pyplot(fig_live, use_container_width=True)

    if analyze_clicked:
        st.session_state.analysis_result = run_pipeline(
            t_raw=t_raw,
            y_raw=y_raw,
            stride=st.session_state.stride,
            direction=direction,
            region_cfg=region_cfg,
            mode=mode,
        )

    res = st.session_state.analysis_result
    if res is None:
        return

    minimal_keys = [
        "T1_peak_time",
        "T1_peak_height",
        "CL_peak_time",
        "CL_peak_height",
        "T_C_ratio_universal_baseline",
        "T_C_ratio_width_baseline",
        "strip_class_universal",
        "strip_class_width",
    ]
    minimal_row = {k: res.get(k, np.nan) for k in minimal_keys}
    full_row = flatten_result(res)

    with col_right:
        st.subheader("Results")
        st.dataframe(pd.DataFrame([minimal_row]), use_container_width=True)
        if show_more:
            st.dataframe(pd.DataFrame([full_row]), use_container_width=True)

        st.subheader("Analysis Plots")
        title_prefix = (
            f"Mode={mode} | direction={direction} | stride={st.session_state.stride} | "
            f"T/Cu={res['T_C_ratio_universal_baseline']:.3f} | T/Cw={res['T_C_ratio_width_baseline']:.3f} |"
        )
        fig1, fig2, fig3 = make_pipeline_plots(res, title_prefix=title_prefix)
        st.pyplot(fig1, use_container_width=True)
        st.pyplot(fig2, use_container_width=True)
        st.pyplot(fig3, use_container_width=True)

        zip_bytes = build_results_zip(
            minimal_row=minimal_row,
            full_row=full_row,
            figs_dict={
                "plot_live.png": fig_live,
                "plot_1_signal_baseline.png": fig1,
                "plot_2_corrected_peaks.png": fig2,
                "plot_3_flat_baseline.png": fig3,
            },
        )
        st.download_button(
            "Download Results",
            data=zip_bytes,
            file_name="strip_analysis_results.zip",
            mime="application/zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pybaselines.whittaker import asls
from scipy.signal import find_peaks, savgol_filter


# ----------------------------- Defaults / Config ------------------------------
ALS_LAMBDA = 1e7
ALS_P = 0.999
SMOOTH_WINDOW = 15
T1_INNER_TRIM_FRACTION = 0.25
WIDTH_DIP_EPS = 0.3
WIDTH_FLAT_STEPS = 3

MODE_OPTIONS = ["AFM (default)", "3 in one", "4 in one"]
STRIDE_OPTIONS = [0, 2, 3, 4]
DEFAULT_STRIDE = 4


# ------------------------------ Data Utilities -------------------------------
def load_data(uploaded_file):
    """Load CSV and return t, selected run signal, and metadata."""
    df = pd.read_csv(uploaded_file)
    df.columns = [str(c).strip() for c in df.columns]

    if "t" not in df.columns:
        raise ValueError("CSV must contain a 't' column.")

    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)

    run_cols = [c for c in df.columns if c.lower().startswith("run")]
    if run_cols:
        run_col = "run1" if "run1" in run_cols else run_cols[0]
    else:
        numeric_cols = [c for c in df.columns if c != "t" and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not numeric_cols:
            raise ValueError("Could not find signal column (run1 or any numeric column).")
        run_col = numeric_cols[0]

    y = pd.to_numeric(df[run_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 5:
        raise ValueError("Too few valid points after NaN removal.")

    return {"df": df, "t": t, "y": y, "run_col": run_col}


def interleave_signal(t, y, stride):
    """Interleave-average signal. stride=0 means raw signal (no interleaving)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if int(stride) == 0:
        return t, y, []

    n_groups = int(max(1, stride))
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

    min_len = min(len(yy) for _, yy in parts)
    if min_len < 3:
        return t, y, []

    t_stack = np.vstack([tt[:min_len] for tt, _ in parts])
    y_stack = np.vstack([yy[:min_len] for _, yy in parts])
    t_avg = np.nanmean(t_stack, axis=0)
    y_avg = np.nanmean(y_stack, axis=0)
    parts_trimmed = [(tt[:min_len], yy[:min_len]) for tt, yy in parts]
    return t_avg, y_avg, parts_trimmed


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
    """SG smooth -> ALS baseline (fit on edge-trimmed SG) -> corrected = baseline - smoothed."""
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
        start_idx, end_idx = 0, n

    fit_idx = np.arange(start_idx, end_idx, dtype=int)
    smoothed_fit = smoothed[start_idx:end_idx]
    baseline_fit, _ = asls(smoothed_fit, lam=lam, p=p)
    baseline = np.interp(np.arange(n), fit_idx, baseline_fit)
    corrected = baseline - smoothed
    return smoothed, baseline, corrected


# ------------------------------ Peak / Width ---------------------------------
def _inner_slice(indices, trim_fraction):
    n = len(indices)
    if n == 0:
        return indices
    trim_n = int(np.floor(float(trim_fraction) * n))
    if n - 2 * trim_n < 1:
        return indices
    return indices[trim_n:n - trim_n]


def find_dip(y, peak_idx, direction):
    """
    Simple slope-based dip detection.
    Rules:
    - Flat region: WIDTH_FLAT_STEPS consecutive |diff| < WIDTH_DIP_EPS -> first flat point
    - Increase after decrease: diff > eps after decreasing trend -> i-1 (or i+1 for left scan)
    - Return None if no clear dip.
    """
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


def _classify_ratio(tc_ratio, cl_present):
    if not cl_present:
        return "INVALID"
    if not np.isfinite(tc_ratio):
        return "INVALID"
    if tc_ratio > 1.1:
        return "NEGATIVE"
    if 0.9 <= tc_ratio <= 1.1:
        return "WEAK POSITIVE"
    return "POSITIVE"


def run_pipeline(t_raw, y_raw, stride, direction, region_cfg):
    """
    Full pipeline:
    Raw -> interleave -> SG -> ALS -> corrected -> region peaks -> width -> ratios
    """
    t_proc, y_proc, parts = interleave_signal(t_raw, y_raw, stride)
    smoothed, baseline, corrected = correct_baseline(y_proc)

    # Region order follows direction.
    if direction == "forward":
        labels = ["CL", "T1"]
    else:
        labels = ["T1", "CL"]

    peaks = []
    regions = []
    for label in labels:
        center = float(region_cfg[label]["center"])
        width = float(region_cfg[label]["width"])
        start = center - width / 2.0
        end = center + width / 2.0
        regions.append((label, start, end))
        peaks.append(_find_peak_in_region(t_proc, corrected, label, start, end))

    found = {p["line_label"]: p for p in peaks if p["peak_found"]}
    t_height = found.get("T1", {}).get("peak_height", np.nan)
    c_height = found.get("CL", {}).get("peak_height", np.nan)
    t_width = found.get("T1", {}).get("peak_width", np.nan)
    c_width = found.get("CL", {}).get("peak_width", np.nan)
    t_width_y = found.get("T1", {}).get("width_y", np.nan)
    c_width_y = found.get("CL", {}).get("width_y", np.nan)
    t_time = found.get("T1", {}).get("peak_time", np.nan)
    c_time = found.get("CL", {}).get("peak_time", np.nan)

    # If T missing, treat as T = 0.
    t_height_safe = 0.0 if not np.isfinite(t_height) else float(t_height)
    t_height_wb = 0.0 if not np.isfinite(t_height) or not np.isfinite(t_width_y) else float(t_height - t_width_y)
    c_height_wb = (c_height - c_width_y) if (np.isfinite(c_height) and np.isfinite(c_width_y)) else np.nan

    tc_u = (t_height_safe / c_height) if (np.isfinite(c_height) and c_height != 0) else np.nan
    tc_w = (t_height_wb / c_height_wb) if (np.isfinite(c_height_wb) and c_height_wb != 0) else np.nan
    cl_present = bool(np.isfinite(c_height) and c_height > 0)

    return {
        "times": t_proc,
        "raw": y_proc,
        "split_series": parts,
        "smoothed": smoothed,
        "baseline": baseline,
        "corrected": corrected,
        "peaks": peaks,
        "regions": regions,
        "T1_peak_time": t_time,
        "T1_peak_height": t_height,
        "CL_peak_time": c_time,
        "CL_peak_height": c_height,
        "T1_peak_width": t_width,
        "CL_peak_width": c_width,
        "T1_width_baseline_y": t_width_y,
        "CL_width_baseline_y": c_width_y,
        "T_C_ratio_universal_baseline": tc_u,
        "T_C_ratio_width_baseline": tc_w,
        "strip_class_universal": _classify_ratio(tc_u, cl_present),
        "strip_class_width": _classify_ratio(tc_w, cl_present),
    }


# ------------------------------- Plot Helpers --------------------------------
def _plot_live_signal(t, y, split_series, regions, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (ts, ys) in enumerate(split_series):
        ax.plot(ts, ys, color="gray", alpha=0.18, lw=0.7, label="Interleaved parts" if i == 0 else None)
    ax.plot(t, y, lw=1.8, color="tab:blue", label="Signal (raw/interleaved avg)")
    for label, s, e in regions:
        ax.axvspan(s, e, alpha=0.14, color="gold")
        ax.text((s + e) / 2, np.nanmax(y), label, ha="center", va="top", fontsize=9, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("signal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def make_pipeline_plots(result, title_prefix=""):
    t = result["times"]
    pmap = {p["line_label"]: p for p in result["peaks"]}

    # Plot 1: Raw + Baseline
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for i, (ts, ys) in enumerate(result["split_series"]):
        ax1.plot(ts, ys, color="gray", alpha=0.18, lw=0.7, label="Interleaved parts" if i == 0 else None)
    ax1.plot(t, result["raw"], "b-", lw=1.2, label="Signal")
    ax1.plot(t, result["smoothed"], "c-", lw=1.0, label="Smoothed")
    ax1.plot(t, result["baseline"], "r-", lw=2.0, label="ALS Baseline")
    ax1.set_title(f"{title_prefix} Signal + Baseline")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    fig1.tight_layout()

    # Plot 2: Corrected + Peaks
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t, result["corrected"], color="green", lw=1.2)
    ax2.axhline(0, color="black", ls="--", alpha=0.5)
    for label, s, e in result["regions"]:
        ax2.axvspan(s, e, alpha=0.15, color="gold")
        ax2.text((s + e) / 2, np.nanmax(result["corrected"]), label, ha="center", va="top", fontsize=9, fontweight="bold")
    for label in ["CL", "T1"]:
        p = pmap.get(label)
        if not p or not p["peak_found"]:
            continue
        pt = p["peak_time"]
        ph = p["peak_height"]
        ax2.plot(pt, ph, "rv", ms=8)
        ax2.annotate(f"{label}:{ph:.0f}", (pt, ph), textcoords="offset points", xytext=(0, 8), ha="center", color="red")
        if p.get("width_found", False):
            xl, xr, wy = p["width_left_time"], p["width_right_time"], p["width_y"]
            ax2.plot([xl, xl], [wy, ph], color="tab:blue", ls="--", lw=1.2)
            ax2.plot([xr, xr], [wy, ph], color="tab:blue", ls="--", lw=1.2)
            ax2.plot([xl, xr], [wy, wy], color="tab:blue", lw=1.4)
            ax2.annotate(f"W={p['peak_width']:.3f}", ((xl + xr) / 2, wy), textcoords="offset points", xytext=(0, -12), ha="center", color="tab:blue")
    ax2.set_title(f"{title_prefix} Corrected Signal + Peaks")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # Plot 3: Flat baseline
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    flat = result["smoothed"] - result["baseline"]
    ax3.plot(t, flat, color="purple", lw=1.2)
    ax3.axhline(0, color="black", ls="--", alpha=0.5)
    ax3.fill_between(t, flat, 0, where=(flat < 0), alpha=0.25, color="orange")
    for _, s, e in result["regions"]:
        ax3.axvspan(s, e, alpha=0.12, color="gold")
    ax3.set_title(f"{title_prefix} Flat Baseline")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    return fig1, fig2, fig3


def build_results_zip(minimal_row, full_row, figs_dict):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report_minimal.csv", pd.DataFrame([minimal_row]).to_csv(index=False))
        zf.writestr("report_full.csv", pd.DataFrame([full_row]).to_csv(index=False))
        for name, fig in figs_dict.items():
            fbuf = BytesIO()
            fig.savefig(fbuf, format="png", dpi=160, bbox_inches="tight")
            fbuf.seek(0)
            zf.writestr(name, fbuf.getvalue())
    buf.seek(0)
    return buf


# ----------------------------------- UI --------------------------------------
st.set_page_config(page_title="LFA Strip Analyzer", layout="wide")
st.title("Lateral Flow Strip Signal Analyzer")

if "mode" not in st.session_state:
    st.session_state.mode = MODE_OPTIONS[0]
if "stride" not in st.session_state:
    st.session_state.stride = DEFAULT_STRIDE
if "direction" not in st.session_state:
    st.session_state.direction = "forward"
if "region_cfg" not in st.session_state:
    st.session_state.region_cfg = {"forward": {}, "backward": {}}
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def _init_direction_regions(t, direction):
    """Initialize region state once; never auto-reset after user edits."""
    if st.session_state.region_cfg.get(direction) and "CL" in st.session_state.region_cfg[direction] and "T1" in st.session_state.region_cfg[direction]:
        return
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    span = max(1e-6, tmax - tmin)
    # Dynamic defaults from signal span (not hardcoded absolute times)
    if direction == "forward":
        defaults = {"CL": (tmin + 0.35 * span, 0.25 * span), "T1": (tmin + 0.72 * span, 0.25 * span)}
    else:
        defaults = {"T1": (tmin + 0.35 * span, 0.25 * span), "CL": (tmin + 0.72 * span, 0.25 * span)}
    st.session_state.region_cfg[direction] = {
        k: {"center": float(v[0]), "width": float(max(1e-6, v[1]))} for k, v in defaults.items()
    }


col_left, col_right = st.columns([1, 4], gap="large")

with col_left:
    with st.expander("Controls", expanded=True):
        st.session_state.mode = st.selectbox("Mode", MODE_OPTIONS, index=MODE_OPTIONS.index(st.session_state.mode))
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.session_state.stride = st.selectbox("Stride", STRIDE_OPTIONS, index=STRIDE_OPTIONS.index(st.session_state.stride))
        st.session_state.direction = st.radio("Direction", ["forward", "backward"], horizontal=True, index=0 if st.session_state.direction == "forward" else 1)
        analyze_clicked = st.button("Analyze", use_container_width=True)
        show_more = st.toggle("Show more", value=False)

if uploaded is None:
    with col_right:
        st.info("Upload a CSV with columns like `t, run1` to start.")
    st.stop()

try:
    data = load_data(uploaded)
except Exception as e:
    with col_right:
        st.error(f"Failed to load file: {e}")
    st.stop()

t_raw = data["t"]
y_raw = data["y"]
run_col = data["run_col"]
direction = st.session_state.direction
_init_direction_regions(t_raw, direction)

# Direction-specific region controls (persisted)
with col_left:
    with st.expander(f"Region Controls ({direction})", expanded=True):
        tmin, tmax = float(np.nanmin(t_raw)), float(np.nanmax(t_raw))
        span = max(1e-6, tmax - tmin)
        for label in ["CL", "T1"]:
            key_c = f"{direction}_{label}_center"
            key_w = f"{direction}_{label}_width"
            if key_c not in st.session_state:
                st.session_state[key_c] = float(st.session_state.region_cfg[direction][label]["center"])
            if key_w not in st.session_state:
                st.session_state[key_w] = float(st.session_state.region_cfg[direction][label]["width"])

            st.session_state[key_c] = st.slider(
                f"{label} center",
                min_value=tmin,
                max_value=tmax,
                value=float(st.session_state[key_c]),
                step=float(span / 500.0),
                key=f"{key_c}_slider",
            )
            st.session_state[key_w] = st.slider(
                f"{label} width",
                min_value=float(span / 200.0),
                max_value=float(span),
                value=float(st.session_state[key_w]),
                step=float(span / 500.0),
                key=f"{key_w}_slider",
            )
            st.session_state.region_cfg[direction][label]["center"] = float(st.session_state[key_c])
            st.session_state.region_cfg[direction][label]["width"] = float(st.session_state[key_w])

# Live signal preview (immediate)
t_live, y_live, parts_live = interleave_signal(t_raw, y_raw, st.session_state.stride)
live_regions = []
for label in ["CL", "T1"]:
    c = st.session_state.region_cfg[direction][label]["center"]
    w = st.session_state.region_cfg[direction][label]["width"]
    live_regions.append((label, c - w / 2.0, c + w / 2.0))

with col_right:
    st.subheader("Live Signal")
    fig_live = _plot_live_signal(
        t_live,
        y_live,
        parts_live,
        live_regions,
        title=f"Mode={st.session_state.mode} | run={run_col} | direction={direction} | stride={st.session_state.stride}",
    )
    st.pyplot(fig_live, use_container_width=True)

if analyze_clicked:
    st.session_state.analysis_result = run_pipeline(
        t_raw=t_raw,
        y_raw=y_raw,
        stride=st.session_state.stride,
        direction=direction,
        region_cfg=st.session_state.region_cfg[direction],
    )

res = st.session_state.analysis_result
if res is None:
    st.stop()

# Minimal output
minimal_keys = [
    "T1_peak_time",
    "T1_peak_height",
    "CL_peak_time",
    "CL_peak_height",
    "T_C_ratio_universal_baseline",
    "T_C_ratio_width_baseline",
    "strip_class_universal",
    "strip_class_width",
]
minimal_row = {k: res.get(k, np.nan) for k in minimal_keys}

with col_right:
    st.subheader("Results")
    st.dataframe(pd.DataFrame([minimal_row]), use_container_width=True)

    if show_more:
        # Full flattened view
        full_row = {}
        for k, v in res.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                full_row[k] = str(v)
            else:
                full_row[k] = v
        st.dataframe(pd.DataFrame([full_row]), use_container_width=True)
    else:
        full_row = {}
        for k, v in res.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                full_row[k] = str(v)
            else:
                full_row[k] = v

    st.subheader("Analysis Plots")
    title_prefix = (
        f"Mode={st.session_state.mode} | direction={direction} | stride={st.session_state.stride} | "
        f"T/Cu={res['T_C_ratio_universal_baseline']:.3f} | T/Cw={res['T_C_ratio_width_baseline']:.3f} | "
    )
    fig1, fig2, fig3 = make_pipeline_plots(res, title_prefix=title_prefix)
    st.pyplot(fig1, use_container_width=True)
    st.pyplot(fig2, use_container_width=True)
    st.pyplot(fig3, use_container_width=True)

    zip_bytes = build_results_zip(
        minimal_row=minimal_row,
        full_row=full_row,
        figs_dict={
            "plot_live.png": fig_live,
            "plot_1_signal_baseline.png": fig1,
            "plot_2_corrected_peaks.png": fig2,
            "plot_3_flat_baseline.png": fig3,
        },
    )
    st.download_button(
        "Download Results",
        data=zip_bytes,
        file_name="strip_analysis_results.zip",
        mime="application/zip",
        use_container_width=True,
    )

