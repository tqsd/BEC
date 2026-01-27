from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import matplotlib as mpl


def apply_matplotlib_style(*, use_tex: bool = True) -> None:
    """
    Apply global Matplotlib styling. Call explicitly from plotter.
    Avoid side effects at import time.
    """
    mpl.rcParams["lines.dash_capstyle"] = "round"
    mpl.rcParams["lines.dash_joinstyle"] = "round"
    mpl.rcParams["axes.grid"] = False  # panels control grid
    mpl.rcParams["text.usetex"] = bool(use_tex)
    if use_tex:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{siunitx}"


@dataclass(frozen=True)
class StyleTheme:
    # Which QD traces to try to plot and their appearance
    qd_order: Sequence[str] = ("P_G", "P_X1", "P_X2", "P_XX")

    qd_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "P_G": "#1f77b4",
            "P_X1": "#ff7f0e",
            "P_X2": "#2ca02c",
            "P_XX": "#d62728",
        }
    )
    qd_widths: Dict[str, float] = field(
        default_factory=lambda: {
            "P_G": 2.2,
            "P_X1": 2.2,
            "P_X2": 2.2,
            "P_XX": 2.2,
        }
    )
    qd_styles: Dict[str, str] = field(
        default_factory=lambda: {
            "P_G": "-",
            "P_X1": "-",
            "P_X2": "--",
            "P_XX": "-.",
        }
    )
    qd_alphas: Dict[str, float] = field(
        default_factory=lambda: {
            "P_G": 1.0,
            "P_X1": 1.0,
            "P_X2": 1.0,
            "P_XX": 1.0,
        }
    )
    qd_labels_tex: Dict[str, str] = field(
        default_factory=lambda: {
            "P_G": r"$P_G$",
            "P_X1": r"$P_{X1}$",
            "P_X2": r"$P_{X2}$",
            "P_XX": r"$P_{XX}$",
        }
    )

    # Palettes for mapping arbitrary output labels -> colors
    palette_outputs: Sequence[str] = (
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    )

    palette_inputs: Sequence[str] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    )


def default_theme() -> StyleTheme:
    return StyleTheme()


def build_color_map(
    labels: Iterable[str], palette: Sequence[str], offset: int = 0
) -> Dict[str, str]:
    """
    Deterministic label->color mapping. Preserves first-seen order.
    """
    uniq: List[str] = []
    seen = set()
    for x in labels:
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    if not uniq:
        return {}

    out: Dict[str, str] = {}
    n = len(palette)
    for i, lbl in enumerate(uniq):
        out[lbl] = palette[(offset + i) % max(n, 1)]
    return out
