from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from bec.quantum_dot.enums import TransitionPair
from bec.units import magnitude

try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box

    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


@dataclass(frozen=True)
class ReportStyle:
    use_color: bool = True
    use_unicode: bool = True
    width: Optional[int] = None
    # layout controls
    columns: bool = True
    # if None -> pick based on width
    columns_min_width: int = 140
    columns_equal: bool = True
    columns_expand: bool = True
    columns_padding: tuple[int, int] = (0, 2)  # (vertical, horizontal)


def has_rich() -> bool:
    return _HAS_RICH


def _auto_width(style: ReportStyle) -> int | None:
    """
    Determine console width:
      - if style.width provided: use it
      - else: use Console().size.width
    Returns None if rich unavailable.
    """
    if not _HAS_RICH:
        return None
    if style.width is not None:
        return int(style.width)
    try:
        return int(Console().size.width)
    except Exception:
        return None


def render_to_text(derived, *, style: ReportStyle) -> str:
    if not _HAS_RICH:
        return derived.report()

    width = _auto_width(style)
    console = Console(record=True, width=width)
    console.print(build_panel(derived, style=style, console=console))
    return console.export_text()


def print_report(derived, *, style: ReportStyle) -> None:
    if not _HAS_RICH:
        print(derived.report())
        return

    width = _auto_width(style)
    console = Console(width=width)
    console.print(build_panel(derived, style=style, console=console))


def build_panel(derived, *, style: ReportStyle, console: Console | None = None):
    """
    Returns a Rich renderable (Panel) that contains your report tables.
    Uses Columns layout when enabled and wide enough; otherwise vertical stacking.
    """
    border = box.ROUNDED if style.use_unicode else box.ASCII

    tables = [
        _energies_table(derived, style=style),
        _polarization_table(derived, style=style),
        _transitions_table(derived, style=style),
        _rates_table(derived, style=style),
        _env_table(derived, style=style),
        _modes_table(derived, style=style),
        _hamiltonians_table(derived, style=style),
    ]

    if console is None:
        width = _auto_width(style)
        console = Console(width=width)

    use_cols = bool(style.columns)
    if style.columns_min_width is not None and console.size.width < int(
        style.columns_min_width
    ):
        use_cols = False

    if use_cols:
        body = Columns(
            tables,
            equal=style.columns_equal,
            expand=style.columns_expand,
            padding=style.columns_padding,
        )
    else:
        body = Group(*tables)

    return Panel(body, title="DerivedQD", box=border, padding=(1, 1))


def _energies_table(derived, *, style: ReportStyle):
    border = box.SIMPLE_HEAVY if style.use_unicode else box.ASCII
    t = Table(title="Energies", box=border, show_lines=False)
    t.add_column("Level", style="bold" if style.use_color else "")
    t.add_column("Energy (eV)", justify="right")

    E = derived.energies
    t.add_row("X1", f"{magnitude(E['X1'], 'eV'):.6f}")
    t.add_row("X2", f"{magnitude(E['X2'], 'eV'):.6f}")
    t.add_row("XX", f"{magnitude(E['XX'], 'eV'):.6f}")

    es = getattr(derived.qd, "energy_structure", None)
    if es is not None:
        t.add_section()
        t.add_row(
            "exciton_center",
            f"{
                magnitude(es.exciton_center, 'eV'):.6f}",
        )
        t.add_row("fss", f"{magnitude(es.fss, 'eV'):.6e}")
        t.add_row("binding", f"{magnitude(es.binding, 'eV'):.6e}")

    return t


def _polarization_table(derived, *, style: ReportStyle):
    border = box.SIMPLE_HEAVY if style.use_unicode else box.ASCII
    t = Table(title="Polarization", box=border, show_lines=False)
    t.add_column("Item", style="bold" if style.use_color else "")
    t.add_column("Value", justify="right")

    pol = getattr(derived.qd, "polarization", None)
    if pol is None:
        t.add_row("status", "none")
        return t

    t.add_row("theta [rad]", f"{float(pol.theta):.4f}")
    t.add_row("phi   [rad]", f"{float(pol.phi):.4f}")
    t.add_row("Omega [eV]", f"{magnitude(pol.Omega, 'eV'):.6e}")

    ep = np.asarray(pol.e_plus_hv(), dtype=complex).reshape(2)
    em = np.asarray(pol.e_minus_hv(), dtype=complex).reshape(2)

    def _c(z: complex) -> str:
        return f"{z.real:.3g}{z.imag:+.3g}j"

    if style.use_color:
        t.add_row(
            "e_plus(H,V)",
            f"[cyan]{_c(ep[0])}[/cyan], [magenta]{_c(ep[1])}[/magenta]",
        )
        t.add_row(
            "e_minus(H,V)",
            f"[cyan]{_c(em[0])}[/cyan], [magenta]{_c(em[1])}[/magenta]",
        )
    else:
        t.add_row("e_plus(H,V)", f"{_c(ep[0])}, {_c(ep[1])}")
        t.add_row("e_minus(H,V)", f"{_c(em[0])}, {_c(em[1])}")

    return t


def _rates_table(derived, *, style: ReportStyle):
    border = box.SIMPLE_HEAVY if style.use_unicode else box.ASCII
    t = Table(title="Collapse Rates", box=border)
    t.add_column("Key", style="bold" if style.use_color else "")
    t.add_column("rate [1/s]", justify="right")

    rates = getattr(derived, "rates", {}) or {}
    if not rates:
        t.add_row("status", "none")
        return t

    for k in sorted(rates.keys()):
        val = rates[k]
        t.add_row(str(k), f"{float(val.to('1/s').magnitude):.3e}")
    return t


def _transitions_table(derived, *, style: ReportStyle):
    border = box.SIMPLE_HEAVY if style.use_unicode else box.ASCII
    t = Table(title="Transitions", box=border)
    t.add_column("Transition")
    t.add_column("omega [rad/s]", justify="right")
    for tp in (TransitionPair.G_X1, TransitionPair.G_X2, TransitionPair.G_XX):
        t.add_row(tp.value, f"{magnitude(derived.omega(tp), 'rad/s'):.3e}")
    return t


def _env_table(derived, *, style: ReportStyle):
    border = box.SIMPLE_HEAVY if style.use_unicode else box.ASCII
    t = Table(title="Environment", box=border)
    t.add_column("Item")
    t.add_column("Value", justify="right")

    t.add_row("polaron_B", f"{float(derived.polaron_B):.6f}")

    P = getattr(derived, "phonon_params", None)
    if P is not None:
        t.add_row("phonon_model", str(getattr(P, "model", "none")))
        t.add_row(
            "polaron_renorm",
            str(bool(getattr(P, "enable_polaron_renorm", False))),
        )

        try:
            t.add_row(
                "T [K]",
                f"{magnitude(
                    getattr(P, 'temperature_K'), 'K'):.2f}",
            )
        except Exception:
            pass

        try:
            t.add_row(
                "alpha [s^2]",
                f"{float(getattr(P, 'alpha_s2', getattr(P, 'alpha', 0.0))):.3e}",
            )
        except Exception:
            pass

        try:
            t.add_row(
                "omega_c [rad/s]",
                f"{magnitude(getattr(P, 'omega_c_rad_s'), 'rad/s'):.3e}",
            )
        except Exception:
            pass

        for name in ("phi_G", "phi_X", "phi_XX"):
            if hasattr(P, name):
                t.add_row(name, f"{float(getattr(P, name)):.3g}")

    out = getattr(derived, "phonon_outputs", None)
    if out is not None:
        rates = getattr(out, "rates", {}) or {}
        for k, v in rates.items():
            key = k.value if hasattr(k, "value") else str(k)
            try:
                t.add_row(key, f"{magnitude(v, '1/s'):.3e}")
            except Exception:
                t.add_row(key, str(v))

    return t


def _modes_table(derived, *, style: ReportStyle):
    border = box.SIMPLE_HEAVY if style.use_unicode else box.ASCII

    mr = getattr(derived.qd, "mode_registry", None)
    res = getattr(mr, "resolution", None)
    res_s = (
        str(res.value)
        if hasattr(res, "value")
        else (str(res) if res is not None else "unknown")
    )

    rows = getattr(derived, "mode_entries", None) or []

    t = Table(
        title=f"Output Modes [{res_s}, N={len(rows)}]",
        box=border,
        show_lines=False,
    )
    t.add_column("#", justify="right")
    t.add_column("Channel")
    t.add_column("E (eV)", justify="right")
    t.add_column("λ (nm)", justify="right")

    if not rows:
        t.add_row("-", "none", "-", "-")
        return t

    for r in rows:
        idx = str(r.get("idx", ""))
        kind = r.get("kind", "?")
        pol = r.get("pol", "?")
        chan = f"{kind}:{pol}"

        e = r.get("energy_eV")
        lam = r.get("wavelength_nm")

        e_s = "-" if e is None else f"{float(e):.6f}"
        lam_s = "-" if lam is None else f"{float(lam):.2f}"

        t.add_row(idx, chan, e_s, lam_s)

    return t


def _hamiltonians_table(derived, *, style: ReportStyle):
    border = box.SIMPLE_HEAVY if style.use_unicode else box.ASCII
    t = Table(title="Hamiltonians", box=border, show_lines=False)

    t.add_column("Group", style="bold" if style.use_color else "")
    t.add_column("Label")
    t.add_column("Pretty / meta", overflow="fold")

    # If mixin not installed yet
    if not hasattr(derived, "h_catalog"):
        t.add_row("status", "missing", "DerivedQD has no h_catalog()")
        return t

    # Choose a reasonable default timescale for displaying static coefficients
    # If you have a central simulation config, you can wire this later.

    try:
        cat = derived.h_catalog()
    except Exception as e:
        t.add_row("status", "error", repr(e))
        return t

    static_terms = getattr(cat, "static_terms", []) or []
    detuning_basis = getattr(cat, "detuning_basis", []) or []
    coherence_basis = getattr(cat, "coherence_basis", []) or []

    # Summary header rows
    t.add_row("static", f"N={len(static_terms)}", f"time_unit_s={0}")
    t.add_row("basis", f"proj N={len(detuning_basis)}", "detuning projectors")
    t.add_row(
        "basis",
        f"coh  N={
            len(coherence_basis)}",
        "coherences |bra><ket|",
    )

    # Show a few representative terms
    def _term_row(group: str, term) -> None:
        label = getattr(term, "label", "?")
        pretty = getattr(term, "pretty", None) or ""
        meta = getattr(term, "meta", {}) or {}
        meta_type = meta.get("type", "")
        tail = pretty if pretty else str(meta_type)
        t.add_row(group, label, tail)

    if static_terms:
        t.add_section()
        for term in static_terms[:6]:
            _term_row("static", term)

    if detuning_basis:
        t.add_section()
        for term in detuning_basis[:6]:
            _term_row("proj", term)

    if coherence_basis:
        t.add_section()
        for term in coherence_basis[:6]:
            _term_row("coh", term)

    return t
