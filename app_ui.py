"""Streamlit UI for strip analysis (UI orchestration only)."""

from datetime import datetime

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
    # Backward compatibility for previously stored label.
    if st.session_state.mode == "AFM (default)":
        st.session_state.mode = "AFM"
    if "stride" not in st.session_state:
        st.session_state.stride = DEFAULT_STRIDE
    if "direction" not in st.session_state:
        st.session_state.direction = "forward"
    if "smooth_window" not in st.session_state:
        st.session_state.smooth_window = DEFAULT_SG_SMOOTHING
    if "region_cfg" not in st.session_state:
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


def _sync_widget_pair(canon_key, source_key, peer_key):
    """Keep slider/number and canonical value in sync."""
    st.session_state[canon_key] = float(st.session_state[source_key])
    st.session_state[peer_key] = float(st.session_state[source_key])


def _sync_numeric_pair(label, min_value, max_value, step, state_key_base, default_value):
    """
    Render slider + number_input synced to one canonical state key.
    Returns current canonical value.
    """
    state_key = state_key_base
    slider_key = f"{state_key_base}|slider"
    number_key = f"{state_key_base}|number"
    slider_widget_key = f"{slider_key}|widget"
    number_widget_key = f"{number_key}|widget"

    if state_key not in st.session_state:
        st.session_state[state_key] = float(default_value)
    if slider_key not in st.session_state:
        st.session_state[slider_key] = float(st.session_state[state_key])
    if number_key not in st.session_state:
        st.session_state[number_key] = float(st.session_state[state_key])
    # Always seed widget values from canonical state BEFORE rendering widgets.
    # This avoids resets on non-widget reruns (e.g. Analyze button clicks).
    st.session_state[slider_widget_key] = float(st.session_state[state_key])
    st.session_state[number_widget_key] = float(st.session_state[state_key])
    c1, c2 = st.columns(2)
    with c1:
        st.slider(
            f"{label} (slider)",
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=slider_widget_key,
            on_change=_sync_widget_pair,
            args=(state_key, slider_widget_key, number_widget_key),
        )
    with c2:
        st.number_input(
            f"{label} (number)",
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=number_widget_key,
            on_change=_sync_widget_pair,
            args=(state_key, number_widget_key, slider_widget_key),
        )

    # Canonical value is updated via on_change callback; mirror it to helper keys.
    curr = float(st.session_state[state_key])
    st.session_state[slider_key] = curr
    st.session_state[number_key] = curr
    return curr


def _render_region_controls(mode, direction, t):
    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    span = max(1e-6, tmax - tmin)
    step = float(span / 500.0)

    cfg = _ensure_mode_direction_regions(t, mode, direction)
    labels = get_regions(mode, direction)

    # Single shared width control for all regions in current mode+direction.
    width_base = f"{mode}|{direction}|region_width"
    default_width = float(cfg[labels[0]]["width"]) if labels else float(span * 0.18)
    if width_base not in st.session_state:
        st.session_state[width_base] = float(default_width)
    shared_width = _sync_numeric_pair(
        label="Region width (shared for all)",
        min_value=float(span / 200.0),
        max_value=float(span),
        step=step,
        state_key_base=width_base,
        default_value=default_width,
    )

    for lbl in labels:
        bg = _region_ui_color(lbl)
        st.markdown(
            f"<div style='background:{bg};padding:6px 8px;border-radius:6px;color:#000;'><b>{lbl} region</b></div>",
            unsafe_allow_html=True,
        )
        # Keep center state isolated per mode+direction+region so values persist when switching.
        center_base = f"{mode}|{direction}|{lbl}|center"
        if center_base not in st.session_state:
            st.session_state[center_base] = float(cfg[lbl]["center"])

        center = _sync_numeric_pair(
            label=f"{lbl} center",
            min_value=tmin,
            max_value=tmax,
            step=step,
            state_key_base=center_base,
            default_value=float(cfg[lbl]["center"]),
        )

        # Edge handling: keep [center - width/2, center + width/2] inside bounds.
        half_w = shared_width / 2.0
        if center - half_w < tmin:
            center = tmin + half_w
        if center + half_w > tmax:
            center = tmax - half_w
        # If width too large, collapse to midpoint clamp.
        if half_w > (tmax - tmin) / 2.0:
            center = (tmin + tmax) / 2.0

        # Re-sync clamped center into both controls.
        st.session_state[center_base] = float(center)
        st.session_state[f"{center_base}|slider"] = float(center)
        st.session_state[f"{center_base}|number"] = float(center)

        cfg[lbl]["center"] = float(center)
        cfg[lbl]["width"] = float(shared_width)

    return cfg


def _select_y_column(df, y_columns, default_y_col):
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


def _sorted_test_labels(labels):
    tests = [str(lbl) for lbl in labels if str(lbl).upper() != "CL"]

    def _test_key(label):
        u = label.upper()
        if u.startswith("T"):
            suffix = u[1:]
            if suffix.isdigit():
                return int(suffix)
        return 999

    return sorted(tests, key=_test_key)


def _classify_binary_ratio(ratio, cl_present):
    """Binary POSITIVE/NEGATIVE label for per-test ratios."""
    if not cl_present or not np.isfinite(ratio):
        return "NEGATIVE"
    return "POSITIVE" if float(ratio) <= 1.1 else "NEGATIVE"


def _safe_token(value):
    s = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s).strip("_")


def _build_dynamic_ratio_outputs(result, labels):
    peaks = result.get("peaks", [])
    peaks_by_label = result.get("peaks_by_label", {})
    cl = peaks_by_label.get("CL", {})
    cl_height = cl.get("peak_height", np.nan)
    cl_width_y = cl.get("width_y", np.nan)
    cl_h_wb = (cl_height - cl_width_y) if (np.isfinite(cl_height) and np.isfinite(cl_width_y)) else np.nan
    cl_present = bool(np.isfinite(cl_height) and cl_height > 0)

    ratio_results = {}
    display_rows = []
    tests = _sorted_test_labels(labels)
    for test_label in tests:
        p = peaks_by_label.get(test_label, {})
        t_height = p.get("peak_height", np.nan)
        t_width_y = p.get("width_y", np.nan)
        t_h_wb = (t_height - t_width_y) if (np.isfinite(t_height) and np.isfinite(t_width_y)) else np.nan

        ratio_u = (t_height / cl_height) if (np.isfinite(t_height) and np.isfinite(cl_height) and cl_height != 0) else np.nan
        ratio_w = (t_h_wb / cl_h_wb) if (np.isfinite(t_h_wb) and np.isfinite(cl_h_wb) and cl_h_wb != 0) else np.nan
        class_u = _classify_binary_ratio(ratio_u, cl_present)
        class_w = _classify_binary_ratio(ratio_w, cl_present)

        key = f"{test_label}/CL"
        ratio_results[key] = {
            "universal": ratio_u,
            "width": ratio_w,
            "class_universal": class_u,
            "class_width": class_w,
        }
        display_rows.append(
            {
                "Test": key,
                "Universal": ratio_u,
                "Width": ratio_w,
                "Class Universal": class_u,
                "Class Width": class_w,
            }
        )

    ratios_vertical_df = pd.DataFrame(display_rows)

    # Full long-form: one row per region with region metrics + ratio/class where applicable.
    full_rows = []
    for p in peaks:
        lbl = str(p.get("line_label", ""))
        ratio_key = f"{lbl}/CL"
        ratio_pack = ratio_results.get(ratio_key, {})
        peak_h = p.get("peak_height", np.nan)
        width_y = p.get("width_y", np.nan)
        height_wb = (peak_h - width_y) if (np.isfinite(peak_h) and np.isfinite(width_y)) else np.nan

        row = {
            "region_name": lbl,
            "peak_found": p.get("peak_found", False),
            "peak_time": p.get("peak_time", np.nan),
            "peak_height": peak_h,
            "peak_width": p.get("peak_width", np.nan),
            "width_left_time": p.get("width_left_time", np.nan),
            "width_right_time": p.get("width_right_time", np.nan),
            "width_baseline_y": width_y,
            "height_width_baseline": height_wb,
            "CL_peak_height": cl_height,
            "CL_width_baseline_y": cl_width_y,
            "CL_height_width_baseline": cl_h_wb,
            "T_C_ratio_universal": ratio_pack.get("universal", np.nan),
            "T_C_ratio_width": ratio_pack.get("width", np.nan),
            "class_universal": ratio_pack.get("class_universal", np.nan),
            "class_width": ratio_pack.get("class_width", np.nan),
        }
        for k, v in ratio_results.items():
            safe = k.replace("/", "_")
            row[f"{safe}_universal"] = v["universal"]
            row[f"{safe}_width"] = v["width"]
            row[f"{safe}_class_universal"] = v["class_universal"]
            row[f"{safe}_class_width"] = v["class_width"]
        full_rows.append(row)

    full_long_df = pd.DataFrame(full_rows)
    return ratio_results, ratios_vertical_df, full_long_df


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
    mode = st.session_state.mode
    direction = st.session_state.direction
    with col_left:
        selected_y_col, y_raw = _select_y_column(df, data["y_columns"], data["default_y_col"])

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

    with col_right:
        analyze_col = st.columns([1, 1, 1])[1]
        with analyze_col:
            analyze_clicked = st.button("Analyze", use_container_width=True, key="analyze_button")

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

    labels = get_regions(mode, direction)
    ratio_results, ratios_vertical_df, full_long_df = _build_dynamic_ratio_outputs(res, labels)
    minimal_row = {
        "mode": mode,
        "direction": direction,
        "y_column": selected_y_col,
        "stride": st.session_state.stride,
        "smooth_window": st.session_state.smooth_window,
    }
    for key, pack in ratio_results.items():
        safe = key.replace("/", "_")
        minimal_row[f"{safe}_universal"] = pack["universal"]
        minimal_row[f"{safe}_width"] = pack["width"]
        minimal_row[f"{safe}_class_universal"] = pack["class_universal"]
        minimal_row[f"{safe}_class_width"] = pack["class_width"]
    full_row = flatten_result(res)

    with col_right:
        st.subheader("Results")
        if ratios_vertical_df.empty:
            st.info("No test-line ratios available for display.")
        else:
            st.table(ratios_vertical_df)
        if show_more:
            st.dataframe(full_long_df, use_container_width=True)

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
            extra_tables={
                "ratios_vertical.csv": ratios_vertical_df,
                "report_full_long_form.csv": full_long_df,
            },
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
            file_name=(
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_results_{_safe_token(mode)}_{_safe_token(selected_y_col)}.zip"
            ),
            mime="application/zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

