from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .labels import ax_label
from .panels import (
    draw_coupling_panel,
    draw_drive_panel,
    draw_outputs_panel,
    draw_pops_panel,
)
from .styles import PlotStyle, apply_style, default_style
from .traces import QDTraces


_TIME_SCALE = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
    "fs": 1e-15,
}


def _choose_time_unit(max_s: float) -> str:
    # Pick a display unit so axis numbers are O(1..1e3-ish).
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


def _available_panels(tr: QDTraces, cfg: "PlotConfig") -> List[str]:
    panels: List[str] = []

    have_drive = bool(getattr(tr, "drives", None))
    have_pops = bool(getattr(tr, "pops", None))
    have_out = bool(getattr(tr, "outputs", None))
    have_cpl = bool(getattr(tr, "coherences", None))

    if cfg.show_drive_panel and have_drive:
        panels.append("drive")
    if cfg.show_pops_panel and have_pops:
        panels.append("pops")
    if cfg.show_outputs_panel and have_out:
        panels.append("out")
    if cfg.show_coupling_panel and have_cpl:
        panels.append("coupling")

    return panels


def _draw_panel(
    ax: Axes,
    name: str,
    *,
    t: np.ndarray,
    tu: str,
    tr: QDTraces,
    cfg: "PlotConfig",
    style: PlotStyle,
    title: Optional[str] = None,
) -> None:
    if name == "drive":
        draw_drive_panel(ax, t=t, drives=tr.drives, style=style)
        if title:
            ax.set_title(title)
    elif name == "pops":
        draw_pops_panel(ax, t=t, pops=tr.pops, style=style)
    elif name == "out":
        draw_outputs_panel(ax, t=t, outputs=tr.outputs, style=style)
    elif name == "coupling":
        draw_coupling_panel(
            ax,
            t=t,
            coherences=tr.coherences,
            drives=tr.drives,
            style=style,
            mode=cfg.coupling_mode,
        )
    else:
        raise ValueError(f"Unknown panel name: {name}")


def _sharey_by_row(axes_grid: Sequence[Sequence[Axes]]) -> None:
    # Make each row share y-limits across columns (left axes only).
    # If some panel creates twinx internally, this still keeps the main y consistent.
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


@dataclass
class PlotConfig:
    time_display: str = "auto"
    title: str = "QD run"

    show_drive_panel: bool = True
    show_pops_panel: bool = True
    show_outputs_panel: bool = True
    show_coupling_panel: bool = True

    # kept for compatibility (drive extraction can use this)
    show_omega_L: bool = True
    coupling_mode: str = "abs"  # "abs" or "reim"

    # Multi-column layout
    ncols: int = 1
    sharex: bool = True
    sharey_by_row: bool = True
    column_titles: Optional[Sequence[str]] = None


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
        raise ValueError("Nothing to plot (no panels selected / no data).")

    t, tu = _time_convert(tr.t_s, cfg.time_display)

    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=style.figsize,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, panels)):
        title = cfg.title if (i == 0 and cfg.title) else None
        _draw_panel(
            ax, name, t=t, tu=tu, tr=tr, cfg=cfg, style=style, title=title
        )

    axes[-1].set_xlabel(ax_label("Time", "t", tu))
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

        # Decide panels for this figure:
        # Use the union of available panels across the chunk, but keep a stable order.
        panel_order = ["drive", "pops", "out", "coupling"]
        present = set()
        for tr in chunk:
            present.update(_available_panels(tr, cfg))
        panels = [p for p in panel_order if p in present]
        if not panels:
            raise ValueError("Nothing to plot (no panels selected / no data).")

        # Choose a single time unit for the whole chunk so sharex makes sense
        all_t_s = (
            np.concatenate(
                [
                    _as_1d_float(tr.t_s)
                    for tr in chunk
                    if len(_as_1d_float(tr.t_s)) > 0
                ]
            )
            if chunk
            else np.array([0.0])
        )
        t_dummy, tu = _time_convert(all_t_s, cfg.time_display)
        scale = float(_TIME_SCALE.get(tu, 1.0))

        nrows = len(panels)
        this_cols = len(chunk)

        fig, axes = plt.subplots(
            nrows,
            this_cols,
            sharex=bool(cfg.sharex),
            sharey=False,
            figsize=(style.figsize[0] * this_cols, style.figsize[1]),
            constrained_layout=True,
        )

        # Normalize axes to a 2D list [row][col]
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

        for col, tr in enumerate(chunk):
            t = _as_1d_float(tr.t_s) / scale

            col_title = None
            if cfg.column_titles and (start + col) < len(cfg.column_titles):
                col_title = cfg.column_titles[start + col]

            for row, name in enumerate(panels):
                ax = axes_grid[row][col]
                title = col_title if (row == 0 and col_title) else None
                _draw_panel(
                    ax,
                    name,
                    t=t,
                    tu=tu,
                    tr=tr,
                    cfg=cfg,
                    style=style,
                    title=title,
                )

        # x-label only on bottom row
        for col in range(this_cols):
            axes_grid[-1][col].set_xlabel(ax_label("Time", "t", tu))

        # Optional y-sharing per row across columns
        if cfg.sharey_by_row and this_cols > 1:
            _sharey_by_row(axes_grid)

        figs.append(fig)

    return figs
