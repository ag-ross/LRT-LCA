"""Shared figure constants and matplotlib helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib as _mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as _mtransforms

# outputs/ sits alongside the scripts in workings/.
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "outputs"

# Figure sizes (Springer full and single column).
MM_PER_INCH:         float = 25.4
FIG_WIDTH_FULL_IN:   float = 174.0 / MM_PER_INCH
FIG_WIDTH_SINGLE_IN: float = 84.0  / MM_PER_INCH

# Layout margins (inches); axes height is the reference.
AX_H_IN: float = 1.85
MAR_L:   float = 0.55
MAR_R:   float = 0.06
MAR_T:   float = 0.25
MAR_B:   float = 0.38
GAP_H:   float = 0.40
GAP_W:   float = 0.55

# Typography.
FONT_SIZE_BODY:        int = 6
FONT_SIZE_TICK:        int = 5
FONT_SIZE_PANEL_LABEL: int = 8
PANEL_LABEL_X:         float = -0.09
PANEL_LABEL_Y:         float =  1.04

# Lines, ticks, grids.
LINE_WIDTH_DATA:   float = 0.5
SPINE_LINEWIDTH:   float = 0.35
TICK_MAJOR_WIDTH:  float = 0.35
TICK_MAJOR_LENGTH: float = 2.5
GRID_ALPHA:        float = 0.12
GRID_LINEWIDTH:    float = 0.35
LINE_ALPHA:        float = 0.70
MARKER_SIZE:       float = 2.35
MARKER_EDGEWIDTH:  float = 0.35
DPI_PNG:           int   = 1000

# Line styles cycling within each sector group.
SECTOR_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# Distinct colours for panels with a few individually identifiable curves.
DISTINCT_COLORS: list = [_mpl.colormaps["tab10"](i) for i in range(10)]

# Marker sets for the second and third linestyle recycles.
CURVE_MARKERS_1: list[str] = ["+", "x", "s", "|", "."]
CURVE_MARKERS_2: list[str] = ["^", "v", "D", "1", "*"]


def apply_rcparams() -> None:
    """Set global rcParams to the project style."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":          FONT_SIZE_BODY,
        "axes.labelsize":     FONT_SIZE_BODY,
        "xtick.labelsize":    FONT_SIZE_TICK,
        "ytick.labelsize":    FONT_SIZE_TICK,
        "legend.fontsize":    FONT_SIZE_BODY,
        "axes.linewidth":     SPINE_LINEWIDTH,
        "xtick.major.width":  TICK_MAJOR_WIDTH,
        "ytick.major.width":  TICK_MAJOR_WIDTH,
        "xtick.major.size":   TICK_MAJOR_LENGTH,
        "ytick.major.size":   TICK_MAJOR_LENGTH,
        "hatch.linewidth":    0.4,
    })


def style_ax(ax: plt.Axes, grid: bool = True) -> None:
    """Hide top and right spines; apply module line widths and tick sizes."""
    for name, spine in ax.spines.items():
        if name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_linewidth(SPINE_LINEWIDTH)
    ax.tick_params(axis="both", which="major",
                   width=TICK_MAJOR_WIDTH, length=TICK_MAJOR_LENGTH,
                   labelsize=FONT_SIZE_TICK)
    if grid:
        ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)


def panel_label(ax: plt.Axes, letter: str) -> None:
    """Bold panel letter at the standard outside-axes position."""
    ax.text(PANEL_LABEL_X, PANEL_LABEL_Y, letter,
            transform=ax.transAxes,
            fontsize=FONT_SIZE_PANEL_LABEL, fontweight="bold",
            va="bottom", ha="right")


def set_layout(fig: plt.Figure, nrows: int, ncols: int, w_in: float) -> None:
    """Resize fig so every axes is exactly AX_H_IN tall."""
    h    = MAR_T + nrows * AX_H_IN + max(nrows - 1, 0) * GAP_H + MAR_B
    ax_w = (w_in - MAR_L - MAR_R - max(ncols - 1, 0) * GAP_W) / ncols
    fig.set_size_inches(w_in, h)
    fig.subplots_adjust(
        left   = MAR_L / w_in,
        right  = 1 - MAR_R / w_in,
        bottom = MAR_B / h,
        top    = 1 - MAR_T / h,
        hspace = (GAP_H / AX_H_IN) if nrows > 1 else 0,
        wspace = (GAP_W / ax_w)    if ncols > 1 else 0,
    )


def curve_style_kw(
    rank: int,
    n_pts: int,
    color,
    n_ls: int | None = None,
) -> dict:
    """ax.plot kwargs distinguishing curves past the linestyle cycle; rank is within-colour."""
    if n_ls is None:
        n_ls = len(SECTOR_LINESTYLES)
    if rank < n_ls:
        return {}
    cycle   = (rank - n_ls) // n_ls
    pos     = (rank - n_ls) % n_ls
    markers = CURVE_MARKERS_1 if cycle == 0 else CURVE_MARKERS_2
    mk      = markers[pos % len(markers)]
    step    = max(1, n_pts // 4)
    offset  = (pos * step // max(n_ls, 1)) % step
    return dict(
        marker          = mk,
        markerfacecolor = "none",
        markeredgecolor = color,
        markersize      = MARKER_SIZE,
        markeredgewidth = MARKER_EDGEWIDTH,
        markevery       = (offset, step),
    )


def t1_vline(ax: plt.Axes, t1: float, color: str = "0.4") -> None:
    """Draw a dashed calibration-horizon line at t1 (years) with an inline label."""
    trans = _mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvline(t1, color=color, linewidth=0.5, linestyle="--", zorder=4)
    ax.text(
        t1 + 0.2, 0.97, f"$t_1 = {t1:.0f}$",
        transform=trans, ha="left", va="top",
        fontsize=plt.rcParams["font.size"], color=color,
    )
