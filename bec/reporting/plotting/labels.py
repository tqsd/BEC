from __future__ import annotations


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


def pretty_key(k: str) -> str:
    # Default mapping for common traces; fall back to raw key.
    m = {
        "pop_G": r"$|\mathrm{G}\rangle$",
        "pop_X1": r"$|\mathrm{X_1}\rangle$",
        "pop_X2": r"$|\mathrm{X_2}\rangle$",
        "pop_XX": r"$|\mathrm{XX}\rangle$",
        "n_GX_H": r"$\langle n_{\mathrm{GX,H}}\rangle$",
        "n_GX_V": r"$\langle n_{\mathrm{GX,V}}\rangle$",
        "n_XX_H": r"$\langle n_{\mathrm{XX,H}}\rangle$",
        "n_XX_V": r"$\langle n_{\mathrm{XX,V}}\rangle$",
    }
    if k.startswith("coh_"):
        # coh_G_X1 -> <G|rho|X1>
        parts = k.split("_", 2)
        if len(parts) == 3:
            a = parts[1]
            b = parts[2]
            return rf"$|\langle {a}|\rho|{b}\rangle|$"
        return rf"$|{k}|$"
    return m.get(k, k)
