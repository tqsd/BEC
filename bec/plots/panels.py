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
mpl.rcParams["text.latex.preamble"] = r"\usepackage{siunitx}"


def ax_label(
    text: str, symbol: str | None = None, si: str | None = None
) -> str:
    r"""
    Build 'Text SYMBOL [unit]' with italic symbol and upright SI unit.
    Example: ax_label('Time', 't', r'\nano\second') -> 'Time $t\,[\si{\\nano\\second}]$'
    """
    if symbol and si:
        return rf"{text} ${symbol}\,[\si{{{si}}}]$"
    if symbol:
        return rf"{text} ${symbol}$"
    if si:
        return rf"{text} $[\si{{{si}}}]$"
    return text


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
        ax.set_ylim(-0.05 * y_max_omega, 1.05 * max(y_max_omega, 1e-12))
        # Force sci notation and move the offset (×10^n) to the RIGHT
        fmt = mticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))  # always show ×10^n
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_offset_position("right")  # <- this moves it
        if is_first_col:
            ax.set_ylabel(r"$\Omega(t)$")
            ax.set_ylabel(ax_label("", r"\Omega(t)", r"rad\per\second"))
        else:
            ax.tick_params(axis="y", labelleft=False)

    if area is not None:
        ax2 = ax.twinx()
        fmt2 = mticker.ScalarFormatter(useMathText=True)
        fmt2.set_powerlimits((0, 0))
        ax2.plot(t, area, ls="--", lw=1.4, alpha=0.9)
        ax2.yaxis.set_major_formatter(fmt2)
        ax2.set_ylim(-0.05 * y_max_area, 1.05 * max(y_max_area, 1e-12))
        # (Optional) also ensure sci notation on the right y-axis
        fmt2 = mticker.ScalarFormatter(useMathText=True)
        fmt2.set_powerlimits((0, 0))
        ax2.yaxis.set_major_formatter(fmt2)
        ax2.yaxis.set_offset_position("right")  # right is natural for ax
        if is_last_col:
            ax2.set_ylabel(ax_label("", r"\int^{t}\Omega(t')\,dt'", r"rad"))
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
        ax.set_ylabel(ax_label("Input", r"\langle N\rangle (in)", r"1"))
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
        ax.plot(
            t,
            y,
            color=theme.qd_colors[lab],
            lw=theme.qd_widths[lab],
            ls=theme.qd_styles[lab],
            alpha=theme.qd_alphas[lab],
        )
    if is_first_col:
        ax.set_ylabel(ax_label("QD Pop", None, r"1"))
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
    y_max: float,
) -> None:
    for k, lbl in enumerate(intrinsic_labels):
        if k < len(out_H):
            lw = max(6.0 - 1 * k, 1.0)
            ax.plot(
                t,
                out_H[k],
                color=color_map[lbl],
                linestyle=(0, (0.01, 4.0)),
                lw=lw,
            )
        if k < len(out_V):
            lw = max(2.0 - 0.2 * k, 1.0)
            ax.plot(
                t, out_V[k], color=color_map[lbl], ls="--", lw=lw, alpha=1.0
            )
    if is_first_col:
        ax.set_ylabel(ax_label("Output", r"\langle N \rangle", r"1"))
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(ax_label("Time", "t", r"\nano\second"))
    ax.set_ylim(-0.05, y_max)
    ax.grid(True, alpha=0.3)


def legend_handles(theme, all_intrin, out_color):
    qd_handles = [
        Line2D(
            [0],
            [0],
            color=theme.qd_colors[l],
            lw=theme.qd_widths[l],
            ls=theme.qd_styles[l],
            label=theme.qd_labels_tex[l],
        )
        for l in theme.qd_labels
    ]
    pol_handles = [
        Line2D(
            [0], [0], color="k", lw=4.0, linestyle=(0, (0.01, 4.0)), label="H"
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
