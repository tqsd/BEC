from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .labels import ax_label
from .panels import (
    draw_drive_panel,
    draw_outputs_panel,
    draw_pops_panel,
)
from .styles import PlotStyle, apply_style, default_style
from .traces import QDTraces
from .twin_registry import clear_twin_axes, iter_axes_for_legend


_TIME_SCALE = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
    "fs": 1e-15,
}


def _choose_time_unit(max_s: float) -> str:
    for u in ("s", "ms", "us", "ns", "ps", "fs"):
        if max_s / _TIME_SCALE[u] >= 1.0:
            return u
    return "fs"


def _as_1d_float(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _time_convert(t_s: np.ndarray, time_display: str) -> Tuple[np.ndarray, str]:
    t_s = _as_1d_float(t_s)
    tmax = float(np.nanmax(t_s)) if t_s.size else 1.0
    if time_display == "auto":
        tu = _choose_time_unit(tmax)
    else:
        tu = str(time_display)
    scale = float(_TIME_SCALE.get(tu, 1.0))
    t = t_s / scale
    return t, tu


@dataclass
class PlotConfig:
    time_display: str = "auto"
    title: str = "QD run"

    show_drive_panel: bool = True
    show_pops_panel: bool = True
    show_outputs_panel: bool = True

    # Multi-column layout
    ncols: int = 1
    sharex: bool = True
    sharey_by_row: bool = True
    column_titles: Optional[Sequence[str]] = None

    # Legend behavior
    legend: bool = True
    legend_max_items: int = 80

    # Fixed gutter on the right for the legend (fraction of figure width)
    legend_gutter: float = 0.22


def _available_panels(tr: QDTraces, cfg: PlotConfig) -> List[str]:
    panels: List[str] = []

    have_drive = bool(getattr(tr, "drives", None))
    have_pops = bool(getattr(tr, "pops", None))
    have_out = bool(getattr(tr, "outputs", None))

    if cfg.show_drive_panel and have_drive:
        panels.append("drive")
    if cfg.show_pops_panel and have_pops:
        panels.append("pops")
    if cfg.show_outputs_panel and have_out:
        panels.append("out")

    return panels


def _draw_panel(
    ax: Axes,
    name: str,
    *,
    t: np.ndarray,
    tu: str,
    tr: QDTraces,
    style: PlotStyle,
    show_right_axis: bool,
) -> None:
    if name == "drive":
        draw_drive_panel(
            ax, t=t, tr=tr, style=style, show_right_axis=show_right_axis
        )
    elif name == "pops":
        draw_pops_panel(ax, t=t, tr=tr, style=style)
    elif name == "out":
        draw_outputs_panel(ax, t=t, tr=tr, style=style)
    else:
        raise ValueError("Unknown panel name: %s" % name)


def _sharey_by_row(axes_grid: Sequence[Sequence[Axes]]) -> None:
    nrows = len(axes_grid)
    if nrows == 0:
        return
    ncols = len(axes_grid[0])
    if ncols == 0:
        return

    for r in range(nrows):
        y0 = None
        y1 = None
        for c in range(ncols):
            ax = axes_grid[r][c]
            lo, hi = ax.get_ylim()
            if y0 is None:
                y0, y1 = lo, hi
            else:
                y0 = min(y0, lo)
                y1 = max(y1, hi)
        if y0 is None or y1 is None:
            continue
        for c in range(ncols):
            axes_grid[r][c].set_ylim(y0, y1)


def _collect_legend_items(axes: Sequence[Axes], cfg: PlotConfig):
    """
    Collect legend items grouped by artist gid.

    Expected gids (from panels.py):
      legend:drives
      legend:chirp
      legend:pops
      legend:outputs
    """
    if not cfg.legend:
        return None

    groups = {
        "drives": [],
        "chirp": [],
        "pops": [],
        "outputs": [],
    }

    seen = set()

    for ax in axes:
        for a in iter_axes_for_legend(ax):
            handles, labels = a.get_legend_handles_labels()
            for h, lab in zip(handles, labels):
                lab = str(lab).strip()
                if not lab or lab.startswith("_"):
                    continue
                if lab in seen:
                    continue

                gid = getattr(h, "get_gid", lambda: None)()
                if isinstance(gid, str) and gid.startswith("legend:"):
                    group = gid.split("legend:", 1)[1]
                else:
                    group = None

                if group in groups:
                    groups[group].append((h, lab))
                    seen.add(lab)

    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        return None

    return groups


def _add_legend_gutter(
    fig: plt.Figure, style: PlotStyle, cfg: PlotConfig, legend_groups
) -> None:
    if legend_groups is None:
        return

    gutter = float(cfg.legend_gutter)
    gutter = max(0.10, min(0.40, gutter))

    left = 1.0 - gutter + 0.02
    width = gutter - 0.03

    ax_leg = fig.add_axes([left, 0.08, width, 0.84])
    ax_leg.axis("off")

    handles: List[object] = []
    labels: List[str] = []

    def header(text: str):
        h = plt.Line2D([], [], linestyle="none")
        handles.append(h)
        labels.append(text)

    def spacer():
        h = plt.Line2D([], [], linestyle="none")
        handles.append(h)
        labels.append("")

    order = [
        ("drives", "Drives"),
        ("chirp", "Chirp"),
        ("pops", "QD populations"),
        ("outputs", "Output expectations"),
    ]

    for key, title in order:
        items = legend_groups.get(key)
        if not items:
            continue

        header(r"\textbf{%s}" % title)
        for h, lab in items:
            handles.append(h)
            labels.append(lab)
        spacer()

    leg = ax_leg.legend(
        handles,
        labels,
        loc="center left",
        frameon=False,
        handlelength=2.2,
        handletextpad=0.8,
        borderpad=0.0,
        labelspacing=0.6,
    )

    for text in leg.get_texts():
        if text.get_text().startswith(r"\textbf"):
            text.set_fontsize(text.get_fontsize() * 1.05)


def plot_qd_run(
    tr: QDTraces,
    *,
    cfg: Optional[PlotConfig] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    cfg = cfg or PlotConfig()
    style = style or default_style()
    apply_style(style)

    panels = _available_panels(tr, cfg)
    if not panels:
        raise ValueError("Nothing to plot.")

    t, tu = _time_convert(tr.t_s, cfg.time_display)

    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=style.figsize,
        constrained_layout=False,
    )
    if len(panels) == 1:
        axes = [axes]

    for ax in axes:
        clear_twin_axes(ax)

    for ax, name in zip(axes, panels):
        _draw_panel(
            ax,
            name,
            t=t,
            tu=tu,
            tr=tr,
            style=style,
            show_right_axis=True,
        )

    if cfg.title:
        fig.suptitle(cfg.title)

    axes[-1].set_xlabel(ax_label("Time", "t", tu))

    legend_items = _collect_legend_items(axes, cfg)

    rect_right = (
        1.0 - float(cfg.legend_gutter)
        if (cfg.legend and legend_items is not None)
        else 1.0
    )
    fig.tight_layout(rect=(0.0, 0.0, rect_right, 1.0))

    _add_legend_gutter(fig, style, cfg, legend_items)
    return fig


def plot_qd_runs_grid(
    traces: Sequence[QDTraces],
    *,
    cfg: Optional[PlotConfig] = None,
    style: Optional[PlotStyle] = None,
) -> List[plt.Figure]:
    cfg = cfg or PlotConfig()
    style = style or default_style()
    apply_style(style)

    if len(traces) == 0:
        raise ValueError("traces must be non-empty")

    ncols = int(cfg.ncols) if int(cfg.ncols) > 0 else len(traces)
    ncols = min(ncols, len(traces))

    figs: List[plt.Figure] = []

    for start in range(0, len(traces), ncols):
        chunk = list(traces[start : start + ncols])

        panel_order = ["drive", "pops", "out"]
        present = set()
        for tr in chunk:
            present.update(_available_panels(tr, cfg))
        panels = [p for p in panel_order if p in present]
        if not panels:
            raise ValueError("Nothing to plot.")

        all_t_s = (
            np.concatenate([_as_1d_float(tr.t_s) for tr in chunk])
            if chunk
            else np.array([0.0])
        )
        _t_dummy, tu = _time_convert(all_t_s, cfg.time_display)
        scale = float(_TIME_SCALE.get(tu, 1.0))

        nrows = len(panels)
        this_cols = len(chunk)

        fig, axes = plt.subplots(
            nrows,
            this_cols,
            sharex=bool(cfg.sharex),
            sharey=False,
            figsize=(style.figsize[0], style.figsize[1]),
            constrained_layout=False,
        )

        if nrows == 1 and this_cols == 1:
            axes_grid: List[List[Axes]] = [[axes]]
        elif nrows == 1:
            axes_grid = [list(axes)]
        elif this_cols == 1:
            axes_grid = [[ax] for ax in list(axes)]
        else:
            axes_grid = [list(row) for row in axes]

        if cfg.title:
            fig.suptitle(cfg.title)

        flat_axes: List[Axes] = []
        for row in axes_grid:
            for ax in row:
                clear_twin_axes(ax)
                flat_axes.append(ax)

        for col, tr in enumerate(chunk):
            t = _as_1d_float(tr.t_s) / scale
            is_last_col = col == this_cols - 1

            for row, name in enumerate(panels):
                ax = axes_grid[row][col]
                _draw_panel(
                    ax,
                    name,
                    t=t,
                    tu=tu,
                    tr=tr,
                    style=style,
                    show_right_axis=is_last_col,
                )

        if cfg.column_titles:
            for col in range(this_cols):
                idx = start + col
                if idx < len(cfg.column_titles):
                    title = cfg.column_titles[idx]
                    if title:
                        axes_grid[0][col].set_title(str(title))

        for row in range(nrows):
            for col in range(1, this_cols):
                ax = axes_grid[row][col]
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

        for row in range(nrows - 1):
            for col in range(this_cols):
                axes_grid[row][col].tick_params(labelbottom=False)

        for col in range(this_cols):
            axes_grid[-1][col].set_xlabel(ax_label("Time", "t", tu))

        if cfg.sharey_by_row and this_cols > 1:
            _sharey_by_row(axes_grid)

        legend_items = _collect_legend_items(flat_axes, cfg)

        rect_right = (
            1.0 - float(cfg.legend_gutter)
            if (cfg.legend and legend_items is not None)
            else 1.0
        )
        fig.tight_layout(rect=(0.0, 0.0, rect_right, 1.0))

        _add_legend_gutter(fig, style, cfg, legend_items)
        figs.append(fig)

    return figs
