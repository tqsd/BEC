from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes

from .styles import StyleTheme, default_theme, build_color_map
from .panels import (
    draw_top_panel_classical,
    draw_top_panel_quantum,
    draw_mid_qd,
    draw_bot_outputs,
    legend_handles,
)

# ---------------------------------------------------------------------
# Time-unit helpers
# ---------------------------------------------------------------------

_UNIT_SCALE = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
    "fs": 1e-15,
}


def _choose_time_unit(max_seconds: float) -> str:
    """Pick a display unit so axis numbers land in a reasonable range."""
    for u in ("s", "ms", "us", "ns", "ps", "fs"):
        if max_seconds / _UNIT_SCALE[u] >= 1.0:
            return u
    return "fs"


# ---------------------------------------------------------------------
# Public config
# ---------------------------------------------------------------------


@dataclass
class PlotConfig:
    show_top: bool = True
    figsize: tuple = (9.2, 6.0)
    right_legend_width: float = 0.48
    titles: Optional[List[str]] = None
    filename: Optional[str] = None
    # palette offsets
    inputs_offset: int = 0
    outputs_offset: int = 0
    # time axis control: "auto" or one of "s","ms","us","ns","ps","fs"
    time_display: str = "auto"


# ---------------------------------------------------------------------
# Main grid
# ---------------------------------------------------------------------


@dataclass
class QDPlotGrid:
    """Compose side-by-side columns for one or more QDTraces-like objects."""

    theme: StyleTheme = field(default_factory=default_theme)
    cfg: PlotConfig = field(default_factory=PlotConfig)

    def render(self, traces_list: Iterable[Any]) -> plt.Figure:
        datas = list(traces_list)
        if not (1 <= len(datas) <= 3):
            raise ValueError("QDPlotGrid currently supports 1 to 3 columns.")

        # Build global color maps across all columns
        all_flying = [
            lbl for d in datas for lbl in getattr(d, "flying_labels", [])
        ]
        all_intrin = [
            lbl for d in datas for lbl in getattr(d, "intrinsic_labels", [])
        ]
        fly_color = build_color_map(
            all_flying, self.theme.palette_inputs, self.cfg.inputs_offset
        )
        out_color = build_color_map(
            all_intrin, self.theme.palette_outputs, self.cfg.outputs_offset
        )

        # Decide time display unit (use physical time)
        tmax_sec = max(
            (
                float(
                    np.nanmax(
                        getattr(d, "t", np.array([0.0]))
                        * getattr(d, "time_unit_s", 1.0)
                    )
                )
                for d in datas
            ),
            default=1.0,
        )
        if self.cfg.time_display == "auto":
            time_unit = _choose_time_unit(tmax_sec)
        else:
            time_unit = self.cfg.time_display
        # seconds per display unit
        time_scale = _UNIT_SCALE.get(time_unit, 1.0)

        # Global top-panel limits if any column is classical
        have_classical = any(
            getattr(d, "classical", False)
            and getattr(d, "omega", None) is not None
            for d in datas
        )
        if have_classical:
            # omega in solver units -> convert to rad/s by dividing by time_unit_s
            omega_max = max(
                (
                    float(
                        np.nanmax(d.omega)
                        / max(getattr(d, "time_unit_s", 1.0), 1e-300)
                    )
                    for d in datas
                    if getattr(d, "omega", None) is not None
                ),
                default=1.0,
            )
            # area is already in radians (no scaling)
            area_max = max(
                (
                    float(np.nanmax(d.area))
                    for d in datas
                    if getattr(d, "area", None) is not None
                ),
                default=1.0,
            )
        else:
            omega_max = 1.0
            area_max = 1.0

        # Layout
        ncols = len(datas)
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

        for j, d in enumerate(datas):
            # convert solver time -> physical -> display units
            t_solver = getattr(d, "t", np.array([], dtype=float))
            s_per_unit = float(getattr(d, "time_unit_s", 1.0))
            t_phys = t_solver * s_per_unit
            t_plot = t_phys / time_scale

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

            # -- TOP --
            if self.cfg.show_top:
                is_first = j == 0
                is_last = j == ncols - 1
                if (
                    getattr(d, "classical", False)
                    and getattr(d, "omega", None) is not None
                ):
                    # Convert omega to physical rad/s for plotting
                    omega_solver = d.omega
                    omega_phys = omega_solver / max(s_per_unit, 1e-300)
                    area = d.area  # radians, no scaling
                    draw_top_panel_classical(
                        ax_top,
                        t_plot,
                        omega_phys,
                        area,
                        omega_max,
                        area_max,
                        is_first,
                        is_last,
                    )
                else:
                    draw_top_panel_quantum(
                        ax_top,
                        t_plot,
                        d.flying_labels,
                        d.fly_H,
                        d.fly_V,
                        fly_color,
                        is_first,
                    )
                if self.cfg.titles and j < len(self.cfg.titles):
                    ax_top.set_title(self.cfg.titles[j], fontsize=11)

            # -- MID (QD populations) --
            draw_mid_qd(ax_mid, t_plot, d.qd, self.theme, is_first_col=(j == 0))
            if (
                not self.cfg.show_top
                and self.cfg.titles
                and j < len(self.cfg.titles)
            ):
                ax_mid.set_title(self.cfg.titles[j], fontsize=11)

            # -- BOT (outputs) --
            draw_bot_outputs(
                ax_bot,
                t_plot,
                d.intrinsic_labels,
                d.out_H,
                d.out_V,
                out_color,
                is_first_col=(j == 0),
            )

            # Hide x tick labels except bottom row
            if self.cfg.show_top:
                top_axes[-1].tick_params(axis="x", labelbottom=False)
            ax_mid.tick_params(axis="x", labelbottom=False)

        # Shared legend on the right
        legend_ax = fig.add_subplot(gs[:, -1])
        legend_ax.axis("off")
        handles = legend_handles(self.theme, all_intrin, out_color)
        legend_ax.legend(
            handles=handles, loc="center", frameon=False, handlelength=2.4
        )

        # Tidy ticks
        all_axes = (
            [*top_axes, *mid_axes, *bot_axes]
            if self.cfg.show_top
            else [*mid_axes, *bot_axes]
        )
        for ax in all_axes:
            ax.locator_params(axis="x", nbins=5)
            ax.locator_params(axis="y", nbins=4)

        # Put a shared x label on the bottom row
        if bot_axes:
            bot_axes[-1].set_xlabel(f"Time ({time_unit})")

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
