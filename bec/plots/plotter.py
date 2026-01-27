from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes

from .types import PlotColumn
from .styles import (
    StyleTheme,
    default_theme,
    build_color_map,
    apply_matplotlib_style,
)
from .panels import (
    draw_top_pulse,
    draw_mid_qd,
    draw_bot_outputs,
    legend_handles,
)


_UNIT_SCALE = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
    "fs": 1e-15,
}
_UNIT_TEX = {
    "s": r"\second",
    "ms": r"\milli\second",
    "us": r"\micro\second",
    "ns": r"\nano\second",
    "ps": r"\pico\second",
    "fs": r"\femto\second",
}


def _choose_time_unit(max_seconds: float) -> str:
    for u in ("s", "ms", "us", "ns", "ps", "fs"):
        if max_seconds / _UNIT_SCALE[u] >= 1.0:
            return u
    return "fs"


@dataclass
class PlotConfig:
    show_top: bool = True
    figsize: Tuple[float, float] = (9.2, 6.0)
    right_legend_width: float = 0.48
    filename: Optional[str] = None

    # palette offsets (useful if you compare many figs)
    outputs_offset: int = 0

    # "auto" or "s/ms/us/ns/ps/fs"
    time_display: str = "auto"

    # TeX rendering
    use_tex: bool = True


@dataclass
class QDPlotGrid:
    theme: StyleTheme = field(default_factory=default_theme)
    cfg: PlotConfig = field(default_factory=PlotConfig)

    def render(self, columns: Iterable[PlotColumn]) -> plt.Figure:
        apply_matplotlib_style(use_tex=self.cfg.use_tex)

        cols = list(columns)
        if not (1 <= len(cols) <= 3):
            raise ValueError("QDPlotGrid currently supports 1 to 3 columns.")

        # --- global time scaling ---
        tmax_sec = 0.0
        for c in cols:
            t_phys = np.asarray(c.t_solver, float) * float(c.time_unit_s)
            if t_phys.size:
                tmax_sec = max(tmax_sec, float(np.nanmax(t_phys)))
        unit = (
            _choose_time_unit(tmax_sec)
            if self.cfg.time_display == "auto"
            else self.cfg.time_display
        )
        time_scale = _UNIT_SCALE.get(unit, 1.0)
        time_unit_tex = _UNIT_TEX.get(unit, r"\second")

        # --- global output color map ---
        all_out_labels: List[str] = []
        for c in cols:
            all_out_labels.extend(list(c.outputs_H.keys()))
            all_out_labels.extend(list(c.outputs_V.keys()))
        out_color = build_color_map(
            all_out_labels, self.theme.palette_outputs, self.cfg.outputs_offset
        )

        # --- global y limits for outputs ---
        out_ymax = 0.0
        for c in cols:
            for arr in list(c.outputs_H.values()) + list(c.outputs_V.values()):
                if arr is not None and len(arr):
                    out_ymax = max(out_ymax, float(np.nanmax(arr)))
        out_ymax = max(0.5, 1.10 * out_ymax)

        # --- global y limits for top pulse (if present) ---
        omega_max = 1.0
        area_max = 1.0
        if self.cfg.show_top:
            for c in cols:
                if c.pulse is None:
                    continue
                if c.pulse.omega_solver is not None and len(
                    c.pulse.omega_solver
                ):
                    # omega_phys = omega_solver / time_unit_s
                    omega_phys = np.abs(
                        np.asarray(c.pulse.omega_solver, float)
                        / max(float(c.time_unit_s), 1e-300)
                    )
                    omega_max = max(
                        omega_max, float(np.nanmax(np.abs(omega_phys)))
                    )
                if c.pulse.area_rad is not None and len(c.pulse.area_rad):
                    area_max = max(
                        area_max, float(np.nanmax(np.abs(c.pulse.area_rad)))
                    )

        # --- layout ---
        ncols = len(cols)
        nrows = 3 if self.cfg.show_top else 2

        fig = plt.figure(figsize=self.cfg.figsize, constrained_layout=True)
        fig.set_constrained_layout_pads(
            w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02
        )

        gs = gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols + 1,
            width_ratios=[1] * ncols + [self.cfg.right_legend_width],
            figure=fig,
        )

        top_axes: List[Axes] = []
        mid_axes: List[Axes] = []
        bot_axes: List[Axes] = []

        master_x = None

        for j, c in enumerate(cols):
            # time: solver -> physical -> display
            t_solver = np.asarray(c.t_solver, float)
            t_phys = t_solver * float(c.time_unit_s)
            t_plot = t_phys / float(time_scale)

            if self.cfg.show_top:
                ax_top = fig.add_subplot(gs[0, j], sharex=master_x)
                top_axes.append(ax_top)
                if master_x is None:
                    master_x = ax_top

            mid_row = 1 if self.cfg.show_top else 0
            ax_mid = fig.add_subplot(gs[mid_row, j], sharex=master_x)
            ax_bot = fig.add_subplot(
                gs[(2 if self.cfg.show_top else 1), j], sharex=master_x
            )
            mid_axes.append(ax_mid)
            bot_axes.append(ax_bot)

            # --- TOP ---
            if self.cfg.show_top:
                is_first = j == 0
                is_last = j == ncols - 1
                if c.pulse is not None and c.pulse.omega_solver is not None:
                    omega_phys = np.asarray(c.pulse.omega_solver, float) / max(
                        float(c.time_unit_s), 1e-300
                    )
                    omega_phys_plot = np.abs(omega_phys)
                    area_rad_plot = None
                    if c.pulse.area_rad is not None:
                        area_rad_plot = np.abs(
                            np.asarray(c.pulse.area_rad, float)
                        )
                    draw_top_pulse(
                        ax_top,
                        t_plot,
                        omega_phys=omega_phys_plot,
                        area_rad=area_rad_plot,
                        y_max_omega=omega_max,
                        y_max_area=area_max,
                        is_first_col=is_first,
                        is_last_col=is_last,
                    )
                else:
                    # If no top content, keep panel but make it unobtrusive
                    ax_top.axis("off")

                ax_top.set_title(c.title, fontsize=11)

            # --- MID ---
            draw_mid_qd(
                ax_mid, t_plot, qd=c.qd, theme=self.theme, is_first_col=(j == 0)
            )
            if not self.cfg.show_top:
                ax_mid.set_title(c.title, fontsize=11)

            # --- BOT ---
            draw_bot_outputs(
                ax_bot,
                t_plot,
                outH=c.outputs_H,
                outV=c.outputs_V,
                color_map=out_color,
                is_first_col=(j == 0),
                y_max=out_ymax,
                time_unit_tex=time_unit_tex,
            )

            # ticks: hide top/mid x labels
            if self.cfg.show_top:
                ax_top.tick_params(axis="x", labelbottom=False)
            ax_mid.tick_params(axis="x", labelbottom=False)

        # --- legend panel ---
        legend_ax = fig.add_subplot(gs[:, -1])
        legend_ax.axis("off")
        handles = legend_handles(self.theme, all_out_labels, out_color)
        legend_ax.legend(
            handles=handles, loc="center", frameon=False, handlelength=2.4
        )

        # tidy tick density
        all_axes = (
            [*top_axes, *mid_axes, *bot_axes]
            if self.cfg.show_top
            else [*mid_axes, *bot_axes]
        )
        for ax in all_axes:
            ax.locator_params(axis="x", nbins=5)
            ax.locator_params(axis="y", nbins=4)

        if self.cfg.filename:
            fig.savefig(
                self.cfg.filename,
                dpi=(
                    300
                    if str(self.cfg.filename).lower().endswith(".png")
                    else None
                ),
                bbox_inches="tight",
            )
        return fig
