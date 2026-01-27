from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from bec.quantum_dot.enums import TransitionPair
from bec.units import magnitude

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


@dataclass(frozen=True)
class ReportStyle:
    use_color: bool = True
    use_unicode: bool = True
    width: Optional[int] = None


def has_rich() -> bool:
    return _HAS_RICH


def render_to_text(derived, *, style: ReportStyle) -> str:
    if not _HAS_RICH:
        return derived.report()

    console = Console(record=True, width=style.width)
    console.print(build_panel(derived, style=style))
    return console.export_text()


def print_report(derived, *, style: ReportStyle) -> None:
    if not _HAS_RICH:
        print(derived.report())
        return
    console = Console(width=style.width)
    console.print(build_panel(derived, style=style))


def build_panel(derived, *, style: ReportStyle):
    border = box.ROUNDED if style.use_unicode else box.ASCII

    tables = [
        _energies_table(derived, style=style),
        _polarization_table(derived, style=style),
        _transitions_table(derived, style=style),
        _rates_table(derived, style=style),
        _env_table(derived, style=style),
    ]
    # simple vertical stacking
    from rich.console import Group

    return Panel(Group(*tables), title="DerivedQD", box=border, padding=(1, 1))


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

    # Scalars
    t.add_row("theta [rad]", f"{float(pol.theta):.4f}")
    t.add_row("phi   [rad]", f"{float(pol.phi):.4f}")
    t.add_row("Omega [eV]", f"{magnitude(pol.Omega, 'eV'):.6e}")

    # Eigenvectors (H,V components)
    ep = np.asarray(pol.e_plus_hv(), dtype=complex).reshape(2)
    em = np.asarray(pol.e_minus_hv(), dtype=complex).reshape(2)

    def _c(z: complex) -> str:
        # compact complex formatting
        return f"{z.real:.3g}{z.imag:+.3g}j"

    if style.use_color:
        t.add_row(
            "e_plus(H,V)",
            f"[cyan]{
                _c(ep[0])}[/cyan], [magenta]{_c(ep[1])}[/magenta]",
        )
        t.add_row(
            "e_minus(H,V)",
            f"[cyan]{
                _c(em[0])}[/cyan], [magenta]{_c(em[1])}[/magenta]",
        )
    else:
        t.add_row("e_plus(H,V)", f"{_c(ep[0])}, {_c(ep[1])}")
        t.add_row("e_minus(H,V)", f"{_c(em[0])}, {_c(em[1])}")

    return t


def _rates_table(derived, *, style: ReportStyle):
    border = box.SIMPLE_HEAVY if style.use_unicode else box.ASCII
    t = Table(title="Rates", box=border)
    t.add_column("Key", style="bold" if style.use_color else "")
    t.add_column("rate [1/s]", justify="right")

    rates = getattr(derived, "rates", {}) or {}
    if not rates:
        t.add_row("status", "none")
        return t

    # values are pint quantities already
    for k in sorted(rates.keys()):
        val = rates[k]
        t.add_row(str(k), f"{float(val.to('1/s').magnitude):.3e}")
    return t


def _transitions_table(derived, *, style: ReportStyle):
    t = Table(title="Transitions", box=box.SIMPLE)
    t.add_column("Transition")
    t.add_column("omega [rad/s]", justify="right")
    for tp in (TransitionPair.G_X1, TransitionPair.G_X2, TransitionPair.G_XX):
        t.add_row(tp.value, f"{magnitude(derived.omega(tp), 'rad/s'):.3e}")
    return t


def _env_table(derived, *, style: ReportStyle):
    t = Table(title="Environment", box=box.SIMPLE)
    t.add_column("Item")
    t.add_column("Value", justify="right")

    # Always show B
    t.add_row("polaron_B", f"{float(derived.polaron_B):.6f}")

    P = getattr(derived, "phonon_params", None)
    if P is not None:
        # model / flags
        t.add_row("phonon_model", str(getattr(P, "model", "none")))
        t.add_row(
            "polaron_renorm",
            str(bool(getattr(P, "enable_polaron_renorm", False))),
        )

        # temperature
        try:
            t.add_row(
                "T [K]",
                f"{magnitude(
                getattr(P, 'temperature_K'), 'K'):.2f}",
            )
        except Exception:
            pass

        # spectral density params
        try:
            t.add_row(
                "alpha [s^2]",
                f"{
                      float(getattr(P, 'alpha_s2', getattr(P, 'alpha', 0.0))):.3e}",
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

        # deformation potentials (dimensionless in your model)
        for name in ("phi_G", "phi_X", "phi_XX"):
            if hasattr(P, name):
                t.add_row(name, f"{float(getattr(P, name)):.3g}")

    out = getattr(derived, "phonon_outputs", None)
    if out is not None:
        # show phonon rates if present
        rates = getattr(out, "rates", {}) or {}
        for k, v in rates.items():
            key = k.value if hasattr(k, "value") else str(k)
            try:
                t.add_row(key, f"{magnitude(v, '1/s'):.3e}")
            except Exception:
                t.add_row(key, str(v))

    return t
