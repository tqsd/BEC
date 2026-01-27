from __future__ import annotations

from typing import Dict, Optional, Sequence, List

import numpy as np
from matplotlib.axes import Axes
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D

from .styles import StyleTheme


def ax_label(
    text: str, symbol: str | None = None, si: str | None = None
) -> str:
    if symbol and si:
        return rf"{text} ${symbol}\,[\si{{{si}}}]$"
    if symbol:
        return rf"{text} ${symbol}$"
    if si:
        return rf"{text} $[\si{{{si}}}]$"
    return text


def draw_top_pulse(
    ax: Axes,
    t: np.ndarray,
    *,
    omega_phys: Optional[np.ndarray],
    area_rad: Optional[np.ndarray],
    y_max_omega: float,
    y_max_area: float,
    is_first_col: bool,
    is_last_col: bool,
) -> None:
    ax.grid(True, alpha=0.25)

    if omega_phys is not None:
        ax.plot(t, omega_phys, lw=1.6)
        ax.set_ylim(-0.05 * y_max_omega, 1.05 * max(y_max_omega, 1e-15))

        fmt = mticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_offset_position("right")

        if is_first_col:
            ax.set_ylabel(ax_label("", r"\Omega(t)", r"rad\per\second"))
        else:
            ax.tick_params(axis="y", labelleft=False)

    if area_rad is not None:
        ax2 = ax.twinx()
        ax2.plot(t, area_rad, ls="--", lw=1.4, alpha=0.9)

        fmt2 = mticker.ScalarFormatter(useMathText=True)
        fmt2.set_powerlimits((0, 0))
        ax2.yaxis.set_major_formatter(fmt2)
        ax2.set_ylim(-0.05 * y_max_area, 1.05 * max(y_max_area, 1e-15))
        ax2.yaxis.set_offset_position("right")

        if is_last_col:
            ax2.set_ylabel(ax_label("", r"\int^t\Omega(t')dt'", r"rad"))
            ax2.yaxis.set_major_locator(mticker.MaxNLocator(4))
        else:
            ax2.tick_params(axis="y", right=False, labelright=False)
            ax2.spines["right"].set_visible(False)


def draw_mid_qd(
    ax: Axes,
    t: np.ndarray,
    *,
    qd: Dict[str, np.ndarray],
    theme: StyleTheme,
    is_first_col: bool,
) -> None:
    for key in theme.qd_order:
        y = qd.get(key)
        if y is None:
            continue
        ax.plot(
            t,
            y,
            color=theme.qd_colors.get(key, "k"),
            lw=theme.qd_widths.get(key, 2.0),
            ls=theme.qd_styles.get(key, "-"),
            alpha=theme.qd_alphas.get(key, 1.0),
        )

    if is_first_col:
        ax.set_ylabel(ax_label("QD Pop", None, r"1"))
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.grid(True, alpha=0.25)


def draw_bot_outputs(
    ax: Axes,
    t: np.ndarray,
    *,
    outH: Dict[str, np.ndarray],
    outV: Dict[str, np.ndarray],
    color_map: Dict[str, str],
    is_first_col: bool,
    y_max: float,
    time_unit_tex: str,
) -> None:
    labels = list(dict.fromkeys(list(outH.keys()) + list(outV.keys())))

    for k, lbl in enumerate(labels):
        if lbl in outH:
            lw = max(6.0 - 1.0 * k, 1.0)
            ax.plot(
                t,
                outH[lbl],
                color=color_map.get(lbl, "k"),
                linestyle=(0, (0.01, 4.0)),
                lw=lw,
            )
        if lbl in outV:
            lw = max(2.0 - 0.2 * k, 1.0)
            ax.plot(
                t,
                outV[lbl],
                color=color_map.get(lbl, "k"),
                ls="--",
                lw=lw,
                alpha=1.0,
            )

    if is_first_col:
        ax.set_ylabel(ax_label("Output", r"\langle N \rangle", r"1"))
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xlabel(ax_label("Time", "t", time_unit_tex))
    ax.set_ylim(-0.05 * max(y_max, 1e-12), y_max)
    ax.grid(True, alpha=0.25)


def legend_handles(
    theme: StyleTheme, output_labels: Sequence[str], color_map: Dict[str, str]
) -> List[Line2D]:
    qd_handles = []
    for k in theme.qd_order:
        if k not in theme.qd_labels_tex:
            continue
        qd_handles.append(
            Line2D(
                [0],
                [0],
                color=theme.qd_colors.get(k, "k"),
                lw=theme.qd_widths.get(k, 2.0),
                ls=theme.qd_styles.get(k, "-"),
                label=theme.qd_labels_tex.get(k, k),
            )
        )

    pol_handles = [
        Line2D(
            [0], [0], color="k", lw=4.0, linestyle=(0, (0.01, 4.0)), label="H"
        ),
        Line2D([0], [0], color="k", lw=1.0, ls="--", label="V"),
    ]

    header = Line2D([], [], color="none", label="Outputs (colors)")

    out_label_handles = [
        Line2D([0], [0], color=color_map.get(lbl, "k"), lw=2.0, label=str(lbl))
        for lbl in list(dict.fromkeys(output_labels))
    ]

    return qd_handles + pol_handles + [header] + out_label_handles
