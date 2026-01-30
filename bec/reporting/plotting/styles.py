from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib as mpl


@dataclass(frozen=True)
class PlotStyle:
    use_tex: bool = True
    latex_preamble: str = r"\usepackage{siunitx}"
    grid_alpha: float = 0.3
    figsize: tuple = (9.2, 6.0)
    # legend styling
    legend_facecolor: str = "white"
    legend_edgecolor: str = "black"
    legend_framealpha: float = 1.0
    legend_linewidth: float = 0.9

    qd_labels: List[str] = field(default_factory=list)
    qd_labels_tex: Dict[str, str] = field(default_factory=dict)
    qd_colors: Dict[str, str] = field(default_factory=dict)
    qd_styles: Dict[str, object] = field(default_factory=dict)
    qd_widths: Dict[str, float] = field(default_factory=dict)
    qd_alphas: Dict[str, float] = field(default_factory=dict)

    palette_drives: List[str] = field(default_factory=list)
    palette_outputs: List[str] = field(default_factory=list)

    dotted_marker: str = "o"
    dotted_markevery: int = 40
    dotted_markersize: float = 5


def default_style() -> PlotStyle:
    qd_labels = ["pop_G", "pop_X1", "pop_X2", "pop_XX"]
    qd_labels_tex = {
        "pop_G": r"$|\mathrm{G}\rangle$",
        "pop_X1": r"$|\mathrm{X_1}\rangle$",
        "pop_X2": r"$|\mathrm{X_2}\rangle$",
        "pop_XX": r"$|\mathrm{XX}\rangle$",
    }
    qd_colors = {
        "pop_G": "#0072B2",
        "pop_X1": "#D55E00",
        "pop_X2": "#104911",
        "pop_XX": "#222222",
    }
    qd_styles = {
        "pop_G": "-",
        "pop_X1": (0, (2.0, 2.0)),
        "pop_X2": (0, (0.01, 4.0)),
        "pop_XX": "-",
    }
    qd_widths = {
        "pop_G": 1.4,
        "pop_X1": 1.4,
        "pop_X2": 2.6,
        "pop_XX": 1.4,
    }
    qd_alphas = {k: 0.9 for k in qd_labels}

    palette_outputs = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]

    palette_drives = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]

    return PlotStyle(
        use_tex=True,
        latex_preamble=r"\usepackage{siunitx}",
        grid_alpha=0.3,
        figsize=(9.2, 6.0),
        qd_labels=qd_labels,
        qd_labels_tex=qd_labels_tex,
        qd_colors=qd_colors,
        qd_styles=qd_styles,
        qd_widths=qd_widths,
        qd_alphas=qd_alphas,
        palette_outputs=palette_outputs,
        palette_drives=palette_drives,
    )


def apply_style(style: PlotStyle) -> None:
    mpl.rcParams["lines.dash_capstyle"] = "round"
    mpl.rcParams["lines.dash_joinstyle"] = "round"
    mpl.rcParams["text.usetex"] = bool(style.use_tex)
    if style.use_tex:
        mpl.rcParams["text.latex.preamble"] = style.latex_preamble
