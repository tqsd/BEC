from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    from IPython.display import Math, display  # type: ignore

    _HAS_IPY = True
except Exception:
    _HAS_IPY = False


Factor = tuple[int, str]  # (subsystem_index, local_symbol_str)


def _latex_escape(s: str) -> str:
    # Minimal escaping for common label/meta strings.
    # Extend as needed.
    return (
        s.replace("\\", "\\\\")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def qd_symbol_to_latex(sym: str) -> str | None:
    # Projectors: proj_G, proj_X1, proj_X2, proj_XX
    if sym.startswith("proj_"):
        st = sym.split("proj_", 1)[1]
        return r"\vert %s \rangle\langle %s \vert" % (st, st)

    # Directed transitions: "X1_G" meaning SRC->DST encoded earlier, but your
    # qd4_transition_op uses |DST><SRC|. For name "X1_G": src=X1 dst=G -> |G><X1|
    # So render as sigma_{dst,src} = |dst><src|.
    if "_" in sym and sym.count("_") == 1:
        a, b = sym.split("_", 1)
        # a=SRC, b=DST (based on your Transition value string convention)
        src = a
        dst = b
        return r"\hat{\sigma}_{%s,%s}" % (dst, src)

    # Aliases like "t_X1_G"
    if sym.startswith("t_"):
        inner = sym[2:]
        return qd_symbol_to_latex(inner)

    # Special
    if sym == "sx_G_XX":
        return r"\hat{\sigma}_{XX,G} + \hat{\sigma}_{G,XX}"

    return None


def mode_symbol_to_latex(sym: str) -> str | None:
    if sym == "a":
        return r"\hat{a}"
    if sym == "adag":
        return r"\hat{a}^\dagger"
    if sym == "n":
        return r"\hat{n}"
    if sym == "I":
        return r"\mathbb{I}"
    return None


def symbol_to_latex(sym: str) -> str:
    out = qd_symbol_to_latex(sym)
    if out is not None:
        return out
    out = mode_symbol_to_latex(sym)
    if out is not None:
        return out
    # Fallback: print escaped literal
    return r"\mathrm{%s}" % _latex_escape(sym)


def factors_to_latex(factors: Sequence[Factor], dims: Sequence[int]) -> str:
    # factors describes only the non-identity factors.
    # Everything else is implied identity.
    # Example: [(0, "proj_XX"), (1, "adag")] means:
    #   O = proj_XX (on subsystem 0) \otimes adag (on subsystem 1) \otimes I ...
    max_i = len(dims)
    by_i: dict[int, str] = {int(i): str(s) for i, s in factors}

    parts = []
    for i in range(max_i):
        if i in by_i:
            parts.append(symbol_to_latex(by_i[i]))
        else:
            parts.append(r"\mathbb{I}")
    return r" \otimes ".join(parts)


def coeff_to_latex(coeff: Any) -> str:
    # Minimal, robust fallback: use class name or str(coeff).
    if coeff is None:
        return "1"
    # If you later add coeff.to_latex(), we can prefer it here.
    if hasattr(coeff, "to_latex") and callable(getattr(coeff, "to_latex")):
        try:
            return str(coeff.to_latex())
        except Exception:
            pass
    name = coeff.__class__.__name__
    return r"\mathrm{%s}" % _latex_escape(name)


def term_to_latex(term: Any, dims: Sequence[int]) -> str:
    # Expect term has: label, coeff, meta
    label = getattr(term, "label", "") or ""
    meta: Mapping[str, Any] = getattr(term, "meta", {}) or {}

    if "latex" in meta and isinstance(meta["latex"], str):
        op_ltx = meta["latex"]
    elif "factors" in meta:
        factors = meta["factors"]
        op_ltx = factors_to_latex(factors, dims)
    else:
        # No semantic info available; fall back to label
        op_ltx = r"\mathrm{%s}" % _latex_escape(label if label else "op")

    c_ltx = coeff_to_latex(getattr(term, "coeff", None))

    # Render as c(t) * O
    return r"%s \, %s" % (c_ltx, op_ltx)


def display_problem_latex(
    problem: Any, *, which: str = "h", max_terms: int = 50
) -> None:
    if not _HAS_IPY:
        raise RuntimeError("IPython is not available, cannot display LaTeX.")

    dims = list(getattr(problem, "dims"))
    terms = getattr(problem, which + "_terms")
    lines = []
    for k, t in enumerate(terms[: int(max_terms)]):
        lines.append(r"\left[%02d\right]\; %s" % (k, term_to_latex(t, dims)))

    title = {"h": "H(t)", "c": "C", "e": "E"}.get(which, which)
    body = r"\\ ".join(lines) if lines else r"\mathrm{(none)}"

    ltx = r"\begin{aligned} %s &= %s \end{aligned}" % (title, body)
    display(Math(ltx))


def matrix_to_latex(
    M: np.ndarray, *, max_dim: int = 8, tol: float = 1e-12
) -> str:
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("Expected a square matrix")
    n = M.shape[0]
    if n > int(max_dim):
        return r"\mathrm{matrix\ too\ large:\ %d \times %d}" % (n, n)

    def fmt(z: complex) -> str:
        a = float(np.real(z))
        b = float(np.imag(z))
        if abs(a) < tol:
            a = 0.0
        if abs(b) < tol:
            b = 0.0
        if b == 0.0:
            return r"%g" % a
        if a == 0.0:
            return r"%g\, i" % b
        sign = "+" if b >= 0 else "-"
        return r"%g %s %g\, i" % (a, sign, abs(b))

    rows = []
    for i in range(n):
        row = " & ".join(fmt(complex(M[i, j])) for j in range(n))
        rows.append(row)
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"
