def format_transition_label(lbl: str) -> str:
    """
    Turn transition labels like 'G_X1' or 'X1_XX' into LaTeX
    with reversed order, e.g. '$\\mathrm{X}_{1}\\!\\leftrightarrow\\!\\mathrm{G}$'.

    Falls back to upright text if pattern not recognized.
    """
    lbl = lbl.strip()

    def pretty(s: str) -> str:
        # Map states to upright with subscripts
        if s == "G":
            return r"\mathrm{G}"
        if s == "XX":
            return r"\mathrm{XX}"
        if s == "X1":
            return r"\mathrm{X}_{1}"
        if s == "X2":
            return r"\mathrm{X}_{2}"
        if s == "X":
            return r"\mathrm{X}"
        # default: render upright
        return rf"\mathrm{{{s}}}"

    # Split on first underscore only
    if "_" in lbl:
        left, right = lbl.split("_", 1)
        # Reverse order: right ↔ left
        return rf"${pretty(right)}\!\leftrightarrow\!{pretty(left)}$"

    # Fallback: no underscore
    return rf"$\mathrm{{{lbl}}}$"
