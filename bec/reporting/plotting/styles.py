from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import matplotlib as mpl


# ---- Page presets (inches) ----
A4_WIDTH_IN = 8.27
A4_HEIGHT_IN = 11.69

# "Usual" LaTeX-like margins (roughly 25 mm per side => ~6.30 in text width)
A4_TEXT_WIDTH_IN = 6.30
A4_TEXT_HEIGHT_IN = 9.72


@dataclass(frozen=True)
class PlotStyle:
    # Global / LaTeX
    use_tex: bool = True
    latex_preamble: str = r"\usepackage{siunitx}"

    # Figure geometry (inches)
    figsize: Tuple[float, float] = (A4_TEXT_WIDTH_IN, 5.5)

    # Grid / aesthetics
    grid_alpha: float = 0.3

    # Legend styling
    legend_facecolor: str = "white"
    legend_edgecolor: str = "black"
    legend_framealpha: float = 1.0
    legend_linewidth: float = 0.9

    # QD population styling
    qd_labels: List[str] = field(default_factory=list)
    qd_labels_tex: Dict[str, str] = field(default_factory=dict)
    qd_colors: Dict[str, str] = field(default_factory=dict)
    qd_styles: Dict[str, object] = field(default_factory=dict)
    qd_widths: Dict[str, float] = field(default_factory=dict)
    qd_alphas: Dict[str, float] = field(default_factory=dict)

    # Palettes (drives + outputs)
    palette_drives: List[str] = field(default_factory=list)
    palette_outputs: List[str] = field(default_factory=list)

    # Dotted markers (used by some panels)
    dotted_marker: str = "o"
    dotted_markevery: int = 1000
    dotted_markersize: float = 5.0
    marker_every_x = 0.06

    # ---- convenience builders ----

    def with_figsize(self, width_in: float, height_in: float) -> "PlotStyle":
        return replace(self, figsize=(float(width_in), float(height_in)))

    def a4_single_column(self, height_in: float = 5.5) -> "PlotStyle":
        return self.with_figsize(A4_TEXT_WIDTH_IN, float(height_in))

    def a4_two_column(self, height_in: float = 6.0) -> "PlotStyle":
        return self.with_figsize(2.0 * A4_TEXT_WIDTH_IN, float(height_in))

    def scale(self, factor: float) -> "PlotStyle":
        f = float(factor)
        if f <= 0.0:
            raise ValueError("factor must be > 0")
        w, h = self.figsize
        return self.with_figsize(w * f, h * f)


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
        figsize=(A4_TEXT_WIDTH_IN, 5.5),
        qd_labels=qd_labels,
        qd_labels_tex=qd_labels_tex,
        qd_colors=qd_colors,
        qd_styles=qd_styles,
        qd_widths=qd_widths,
        qd_alphas=qd_alphas,
        palette_outputs=palette_outputs,
        palette_drives=palette_drives,
        dotted_marker="o",
        dotted_markevery=40,
        dotted_markersize=5.0,
    )


def apply_style(style: PlotStyle) -> None:
    mpl.rcParams["lines.dash_capstyle"] = "round"
    mpl.rcParams["lines.dash_joinstyle"] = "round"

    mpl.rcParams["text.usetex"] = bool(style.use_tex)
    if style.use_tex:
        mpl.rcParams["text.latex.preamble"] = style.latex_preamble
