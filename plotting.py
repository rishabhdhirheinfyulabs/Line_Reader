"""Plotting utilities for strip analysis app."""

import matplotlib.pyplot as plt
import numpy as np


def _region_color(label):
    label_u = str(label).upper()
    if label_u == "CL":
        return "#CFE8FF"  # very light blue
    return "#FFF3C9"  # very light amber for T-lines


def plot_live_signal(t, y, split_series, regions, title):
    """Live preview plot (raw/interleaved signal + highlighted regions)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (ts, ys) in enumerate(split_series):
        ax.plot(ts, ys, color="gray", alpha=0.18, lw=0.7, label="Interleaved parts" if i == 0 else None)
    ax.plot(t, y, lw=1.8, color="tab:blue", label="Signal (raw/interleaved avg)")

    y_top = np.nanmax(y) if np.isfinite(np.nanmax(y)) else 1.0
    for label, s, e in regions:
        ax.axvspan(s, e, alpha=0.30, color=_region_color(label))
        ax.text((s + e) / 2, y_top, label, ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("signal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def make_pipeline_plots(result, title_prefix=""):
    """Return the 3 final analysis figures."""
    t = result["times"]
    pmap = {p["line_label"]: p for p in result["peaks"]}

    # Plot 1: Signal + baseline
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

    # Plot 2: Corrected + peaks + widths
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t, result["corrected"], color="green", lw=1.2)
    ax2.axhline(0, color="black", ls="--", alpha=0.5)

    ymax = np.nanmax(result["corrected"]) if np.isfinite(np.nanmax(result["corrected"])) else 1.0
    for label, s, e in result["regions"]:
        ax2.axvspan(s, e, alpha=0.30, color=_region_color(label))
        ax2.text((s + e) / 2, ymax, label, ha="center", va="top", fontsize=9, fontweight="bold")

    for label, p in pmap.items():
        if not p.get("peak_found", False):
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
            ax2.annotate(
                f"W={p['peak_width']:.3f}",
                ((xl + xr) / 2, wy),
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                color="tab:blue",
            )

    ax2.set_title(f"{title_prefix} Corrected Signal + Peaks")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # Plot 3: Flat baseline
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    flat = result["smoothed"] - result["baseline"]
    ax3.plot(t, flat, color="purple", lw=1.2)
    ax3.axhline(0, color="black", ls="--", alpha=0.5)
    ax3.fill_between(t, flat, 0, where=(flat < 0), alpha=0.25, color="orange")
    for label, s, e in result["regions"]:
        ax3.axvspan(s, e, alpha=0.24, color=_region_color(label))
    ax3.set_title(f"{title_prefix} Flat Baseline")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    return fig1, fig2, fig3

