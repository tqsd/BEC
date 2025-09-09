import re


def format_transition_label(lbl: str) -> str:
    """
    Turn 'G_X1' or 'X1_XX' into LaTeX like '$G\\leftrightarrow X_{1}$'.
    Falls back to safely-escaped text if pattern not recognized.
    """
    # Normalize
    lbl = lbl.strip()

    def x_sub(s: str) -> str:
        return s.replace("X1", r"X_{1}").replace("X2", r"X_{2}")

    # Common intrinsic labels
    if lbl in ("G_X1", "G_X2"):
        left, right = lbl.split("_", 1)
        return rf"${left}\leftrightarrow {x_sub(right)}$"
    if lbl in ("X1_XX", "X2_XX"):
        left, right = lbl.split("_", 1)
        return rf"${x_sub(left)}\leftrightarrow {right}$"

    # Anything else: escape underscores; render as upright text
    safe = lbl.replace("_", r"\_")
    return rf"$\mathrm{{{safe}}}$"
