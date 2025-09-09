from __future__ import annotations
from .styles import StyleTheme, latex_mode_label
from bec.plots.util import format_transition_label
from typing import Dict, List, Optional
import numpy as np
from matplotlib.axes import Axes
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
import matplotlib as mpl


mpl.rcParams["lines.dash_capstyle"] = "round"
mpl.rcParams["lines.dash_joinstyle"] = "round"  # 'miter' | 'round' | 'bevel'
mpl.rcParams["text.usetex"] = True


def draw_top_panel_classical(
    ax: Axes,
    t: np.ndarray,
    omega: Optional[np.ndarray],
    area: Optional[np.ndarray],
    y_max_omega: float,
    y_max_area: float,
    is_first_col: bool,
    is_last_col: bool,
) -> None:
    ax.grid(True, alpha=0.3)
    if omega is not None:
        ax.plot(t, omega, lw=1.6)
        ax.set_ylim(0, 1.05 * max(y_max_omega, 1e-12))
        if is_first_col:
            ax.set_ylabel(r"$\Omega(t)$")
        else:
            ax.tick_params(axis="y", labelleft=False)

    if area is not None:
        ax2 = ax.twinx()
        ax2.plot(t, area, ls="--", lw=1.4, alpha=0.9)
        ax2.set_ylim(0, 1.05 * max(y_max_area, 1e-12))
        if is_last_col:
            ax2.set_ylabel(r"$\int^{t}\Omega(t')\,dt'$")
            ax2.yaxis.set_major_locator(mticker.MaxNLocator(4))
            ax2.tick_params(axis="y", right=True, labelright=True)
        else:
            ax2.tick_params(axis="y", right=False, labelright=False)
            ax2.spines["right"].set_visible(False)


def draw_top_panel_quantum(
    ax: Axes,
    t: np.ndarray,
    flying_labels: List[str],
    fly_H: List[np.ndarray],
    fly_V: List[np.ndarray],
    color_map: Dict[str, str],
    is_first_col: bool,
) -> None:
    for k, lbl in enumerate(flying_labels):
        if k < len(fly_V):
            ax.plot(
                t, fly_V[k], color=color_map[lbl], ls="--", lw=1.0, alpha=1.0
            )
        if k < len(fly_H):
            ax.plot(
                t,
                fly_H[k],
                color=color_map[lbl],
                lw=2.0,
                linestyle=(0, (0.2, 4.0)),
                alpha=1.0,
            )
    if is_first_col:
        ax.set_ylabel(r"$\langle N\rangle$ (in)")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(True, alpha=0.3)


def draw_mid_qd(
    ax: Axes,
    t: np.ndarray,
    qd_traces: List[np.ndarray],
    theme: StyleTheme,
    is_first_col: bool,
) -> None:
    for lab, y in zip(theme.qd_labels, qd_traces):
        print(
            lab,
            theme.qd_styles[lab],
            theme.qd_widths[lab],
            theme.qd_alphas[lab],
            theme.qd_colors[lab],
        )
        ax.plot(
            t,
            y,
            color=theme.qd_colors[lab],
            lw=theme.qd_widths[lab],
            ls=theme.qd_styles[lab],
            alpha=theme.qd_alphas[lab],
        )
    if is_first_col:
        ax.set_ylabel("QD")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(True, alpha=0.3)


def draw_bot_outputs(
    ax: Axes,
    t: np.ndarray,
    intrinsic_labels: List[str],
    out_H: List[np.ndarray],
    out_V: List[np.ndarray],
    color_map: Dict[str, str],
    is_first_col: bool,
) -> None:
    for k, lbl in enumerate(intrinsic_labels):
        if k < len(out_H):
            ax.plot(
                t,
                out_H[k],
                color=color_map[lbl],
                linestyle=(0, (0.01, 4.0)),
                lw=2.0,
            )
        if k < len(out_V):
            ax.plot(
                t, out_V[k], color=color_map[lbl], ls="--", lw=1.0, alpha=1.0
            )
    if is_first_col:
        ax.set_ylabel(r"$\langle N\rangle$ (out)")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(r"Time ($ns$)")
    ax.grid(True, alpha=0.3)


def legend_handles(theme, all_intrin, out_color):
    qd_handles = [
        Line2D(
            [0],
            [0],
            color=theme.qd_colors[l],
            lw=theme.qd_widths[l],
            ls=theme.qd_styles[l],
            label=l,
        )
        for l in theme.qd_labels
    ]
    pol_handles = [
        Line2D(
            [0], [0], color="k", lw=2.0, linestyle=(0, (0.01, 4.0)), label="H"
        ),
        Line2D([0], [0], color="k", lw=1.0, ls="--", label="V"),
    ]
    outputs_header = Line2D([], [], color="none", label="Outputs (colors)")

    out_label_handles = [
        Line2D(
            [0],
            [0],
            color=out_color[lbl],
            lw=2.0,
            label=format_transition_label(lbl),
        )
        for lbl in dict.fromkeys(all_intrin)
    ]
    return qd_handles + pol_handles + [outputs_header] + out_label_handles
