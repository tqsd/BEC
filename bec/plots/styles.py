from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

# ---- QD population series (middle panel) ----
QD_LABELS = [
    r"$|G\rangle$",
    r"$|X_1\rangle$",
    r"$|X_2\rangle$",
    r"$|XX\rangle$",
]

QD_COLORS: Dict[str, str] = {
    r"$|G\rangle$": "#0072B2",  # blue
    r"$|X_1\rangle$": "#D55E00",  # vermillion
    r"$|X_2\rangle$": "#104911",  # deep green
    r"$|XX\rangle$": "#222222",  # dark gray
}

QD_STYLES: Dict[str, tuple | str] = {
    r"$|G\rangle$": "-",
    r"$|X_1\rangle$": (0, (2.0, 2.0)),
    r"$|X_2\rangle$": (0, (0.01, 4.0)),
    r"$|XX\rangle$": "-",
}

QD_WIDTHS: Dict[str, float] = {k: 1.4 for k in QD_LABELS}
QD_WIDTHS: Dict[str, float] = {
    r"$|G\rangle$": 1.4,
    r"$|X_1\rangle$": 1.4,
    r"$|X_2\rangle$": 4,
    r"$|XX\rangle$": 1.4,
}
QD_ALPHAS: Dict[str, float] = {k: 0.9 for k in QD_LABELS}
QD_LABELS_TEX: Dict[str, str] = {
    r"$|G\rangle$": r"$|\mathrm{G}\rangle$",
    r"$|X_1\rangle$": r"$|\mathrm{X_1}\rangle$",
    r"$|X_2\rangle$": r"$|\mathrm{X_2}\rangle$",
    r"$|XX\rangle$": r"$|\mathrm{XX}\rangle$",
}

# ---- Palettes for mode coloring (inputs/outputs) ----
PALETTE_A = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
]
PALETTE_B = [
    "#1b9e77",  # teal-green
    "#d95f02",  # orange
    "#7570b3",  # purple-blue
    "#e7298a",  # pink-magenta
    "#66a61e",  # olive green
    "#e6ab02",  # mustard
    "#a6761d",  # brown
]


def build_color_map(
    labels: List[str], palette: List[str], offset: int = 0
) -> Dict[str, str]:
    """Deterministic label→color cycling over a palette."""
    uniq = list(dict.fromkeys(labels))
    n = len(palette)
    return {lbl: palette[(i + offset) % n] for i, lbl in enumerate(uniq)}


def latex_mode_label(lbl: str) -> str:
    s = lbl.replace("X1", r"X_1").replace("X2", r"X_2")
    s = s.replace("<->", r"\leftrightarrow{}").replace("&", r"~\&~")
    return f"${s}$"


@dataclass
class StyleTheme:
    qd_labels: List[str]
    qd_labels_tex: Dict[str, str]
    qd_colors: Dict[str, str]
    qd_styles: Dict[str, tuple | str]
    qd_widths: Dict[str, float]
    qd_alphas: Dict[str, float]
    palette_inputs: List[str]
    palette_outputs: List[str]


def default_theme() -> StyleTheme:
    return StyleTheme(
        qd_labels=QD_LABELS,
        qd_labels_tex=QD_LABELS_TEX,
        qd_colors=QD_COLORS,
        qd_styles=QD_STYLES,
        qd_widths=QD_WIDTHS,
        qd_alphas=QD_ALPHAS,
        palette_inputs=PALETTE_A,
        palette_outputs=PALETTE_B,
    )
