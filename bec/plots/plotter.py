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

# --- Duck-typed view of QDTraces (no hard import needed) ---
# Expected fields on each traces-like object:
#   t: np.ndarray
#   classical: bool
#   flying_labels: List[str]
#   intrinsic_labels: List[str]
#   qd: List[np.ndarray]     (length 4)
#   fly_H, fly_V: List[np.ndarray]
#   out_H, out_V: List[np.ndarray]
#   omega: Optional[np.ndarray]
#   area: Optional[np.ndarray]


@dataclass
class PlotConfig:
    show_top: bool = True
    figsize: tuple = (9.2, 6.0)
    right_legend_width: float = 0.48
    titles: Optional[List[str]] = None
    filename: Optional[str] = None
    # palette offsets (if you want different color cycles)
    inputs_offset: int = 0
    outputs_offset: int = 0


@dataclass
class QDPlotGrid:
    """Compose side-by-side columns for multiple QDTraces objects."""

    theme: StyleTheme = field(default_factory=default_theme)
    cfg: PlotConfig = field(default_factory=PlotConfig)

    def render(self, traces_list: Iterable[Any]) -> plt.Figure:
        datas = list(traces_list)
        if not (1 <= len(datas) <= 3):
            raise ValueError("QDPlotGrid currently supports 1–3 columns.")

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

        # Shared top limits for classical columns
        have_classical = any(
            getattr(d, "classical", False)
            and getattr(d, "omega", None) is not None
            for d in datas
        )
        Ωmax = max(
            (
                float(np.nanmax(d.omega))
                for d in datas
                if getattr(d, "omega", None) is not None
            ),
            default=1.0,
        )
        Amax = max(
            (
                float(np.nanmax(d.area))
                for d in datas
                if getattr(d, "area", None) is not None
            ),
            default=1.0,
        )

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
                    draw_top_panel_classical(
                        ax_top,
                        d.t,
                        d.omega,
                        d.area,
                        Ωmax,
                        Amax,
                        is_first,
                        is_last,
                    )
                else:
                    draw_top_panel_quantum(
                        ax_top,
                        d.t,
                        d.flying_labels,
                        d.fly_H,
                        d.fly_V,
                        fly_color,
                        is_first,
                    )
                if self.cfg.titles and j < len(self.cfg.titles):
                    ax_top.set_title(self.cfg.titles[j], fontsize=11)

            # -- MID (QD pops) --
            draw_mid_qd(ax_mid, d.t, d.qd, self.theme, is_first_col=(j == 0))
            if not self.cfg.show_top:
                if self.cfg.titles and j < len(self.cfg.titles):
                    ax_mid.set_title(self.cfg.titles[j], fontsize=11)

            # -- BOT (outputs) --
            draw_bot_outputs(
                ax_bot,
                d.t,
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

        # Shared legend
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
