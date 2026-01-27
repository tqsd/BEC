from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Union, Any, Mapping, Optional

import numpy as np

from bec.units import QuantityLike, Q, as_quantity, magnitude, c, h, hbar
from bec.quantum_dot.enums import Transition, TransitionPair
from bec.quantum_dot.transitions import DEFAULT_REGISTRY
from bec.quantum_dot.utils.report_rich import ReportStyle, has_rich, render_to_text, print_report as _print_rich
from bec.quantum_dot.parameters.phonons import PhononParams, PhononModelType

HVVec = np.ndarray  # (2,) complex
TrLike = Union[Transition, TransitionPair]


def as_eV(x: Any) -> QuantityLike:
    """Coerce anything energy-like to eV with compatibility checking."""
    return as_quantity(x, "eV")

@dataclass(frozen=True)
class DerivedQD:
    """
    Unitful derived quantities for a QuantumDot instance.

    - All physics quantities returned as pint quantities (via bec.units)
    - Cached where useful
    - Builders can use `.fast_*` helpers to extract floats in desired units
    """

    qd: "QuantumDot"

    # ---- transitions / graph ----

    @cached_property
    def t_registry(self):
        return getattr(self.qd, "transitions", DEFAULT_REGISTRY)

    # ---- energies ----

    @cached_property
    def energies(self) -> Dict[str, QuantityLike]:
        """
        Absolute energies in eV as pint quantities: {"G","X1","X2","XX"}.
        Assumes qd.energy_levels has attributes X1, X2, XX (quantity-like or float in eV).
        """
        el = getattr(self.qd, "energy_structure", None) or self.qd.energy_levels
        return {
            "G": Q(0.0, "eV"),
            "X1": as_eV(el.X1),
            "X2": as_eV(el.X2),
            "XX": as_eV(el.XX),
        }

    @cached_property
    def transition_energy(self) -> Dict[Transition, QuantityLike]:
        """Directed transition energies ΔE as quantities in eV (signed)."""
        E = self.energies
        return {
            Transition.G_X1: (E["X1"] - E["G"]).to("eV"),
            Transition.X1_G: (E["G"] - E["X1"]).to("eV"),
            Transition.G_X2: (E["X2"] - E["G"]).to("eV"),
            Transition.X2_G: (E["G"] - E["X2"]).to("eV"),
            Transition.X1_XX: (E["XX"] - E["X1"]).to("eV"),
            Transition.XX_X1: (E["X1"] - E["XX"]).to("eV"),
            Transition.X2_XX: (E["XX"] - E["X2"]).to("eV"),
            Transition.XX_X2: (E["X2"] - E["XX"]).to("eV"),
            Transition.G_XX: (E["XX"] - E["G"]).to("eV"),
            Transition.XX_G: (E["G"] - E["XX"]).to("eV"),
        }

    @cached_property
    def rates(self) -> Mapping[str, QuantityLike]:
        """
        Best-effort rate dictionary (1/s). Empty if no models configured.
        Keys should be stable strings (or RateKey.value) suitable for logs.
        """
        out: dict[str, QuantityLike] = {}

        dm = getattr(self.qd, "decay_model", None)
        if dm is not None and hasattr(dm, "compute"):
            try:
                # dm.compute() should return Dict[RateKey|str, QuantityLike]
                r = dm.compute()
                for k, v in r.items():
                    key = k.value if hasattr(k, "value") else str(k)
                    out[key] = as_quantity(v, "1/s")
            except Exception:
                pass

        pm = getattr(self.qd, "phonon_model", None)
        if pm is not None and hasattr(pm, "compute_rates"):
            try:
                r = pm.compute_rates()
                for k, v in r.items():
                    key = k.value if hasattr(k, "value") else str(k)
                    out[key] = as_quantity(v, "1/s")
            except Exception:
                pass

        return out

    @cached_property
    def phonon_params(self) -> Optional["PhononParams"]:
        # canonical place: qd.phonon_params
        P = getattr(self.qd, "phonon_params", None)
        if P is not None:
            return P
        # fallback if you only stored it inside the model
        pm = getattr(self.qd, "phonon_model", None)
        return getattr(pm, "_P", None) if pm is not None else None

    @cached_property
    def phonon_outputs(self):
        pm = getattr(self.qd, "phonon_model", None)
        if pm is None or not hasattr(pm, "compute"):
            return None
        try:
            return pm.compute()
        except Exception:
            return None
    # ---- frequency conversion (unitful) ----

    def omega(self, tr: TrLike) -> QuantityLike:
        """
        Angular frequency ω as quantity in rad/s.

        ω = ΔE / ħ
        """
        if isinstance(tr, TransitionPair):
            fwd, _ = self.t_registry.directed(tr)
            tr = fwd

        dE = self.transition_energy[tr].to("J")  # relies on pint having eV->J
        # If your registry doesn't have eV, define it in bec/units.py once.
        return (dE / hbar).to("rad/s")

    def freq(self, tr: TrLike) -> QuantityLike:
        """
        Frequency ν as quantity in Hz.

        ν = ΔE / h
        """
        if isinstance(tr, TransitionPair):
            fwd, _ = self.t_registry.directed(tr)
            tr = fwd

        dE = self.transition_energy[tr].to("J")
        return (dE / h).to("Hz")

    def omega_abs(self, tr: TrLike) -> QuantityLike:
        """|ω| in rad/s (useful for rates, wavelengths, magnitudes)."""
        return abs(self.omega(tr).to("rad/s").magnitude) * Q(1.0, "rad/s")

    def omega_2ph_per_photon(self) -> QuantityLike:
        """Per-photon ω for 2-photon resonance: ω(G->XX)/2."""
        return (self.omega(TransitionPair.G_XX) / 2.0).to("rad/s")

    def wavelength_vacuum(self, tr: TrLike) -> QuantityLike:
        """Vacuum wavelength λ = c / ν as quantity in meters."""
        return (c / self.freq(tr)).to("m")

    # ---- dipole projections / coupling ----

    def mu(self, tr: Transition) -> QuantityLike:
        """Dipole magnitude μ(tr) as quantity in C*m."""
        # assumes DipoleParams.mu(tr) returns QuantityLike or float interpreted as C*m
        if hasattr(self.qd.dipole_params, "mu"):
            return as_quantity(self.qd.dipole_params.mu(tr), "C*m")
        # fallback to your older API mu_Cm(tr)->float
        return Q(float(self.qd.dipole_params.mu_Cm(tr)), "C*m")

    def e_pol_hv(self, tr: Transition) -> HVVec:
        """Normalized polarization vector e_d(tr) in HV basis (dimensionless)."""
        return self.qd.dipole_params.e_pol_hv(tr)

    def drive_projection(self, tr: Transition, E_pol_hv: HVVec) -> complex:
        """
        Complex overlap <e_d(tr)|E_pol> (dimensionless).
        """
        E_pol_hv = np.asarray(E_pol_hv, dtype=complex).reshape(2)
        E_pol_hv = E_pol_hv / np.linalg.norm(E_pol_hv)
        e_d = self.e_pol_hv(tr)
        return complex(np.vdot(e_d, E_pol_hv))

    # ---- phonons ----

    @cached_property
    def polaron_B(self) -> float:
        """Polaron dressing factor <B>(T). Returns 1 if disabled/unavailable."""
        pm = getattr(self.qd, "phonon_model", None)
        if pm is None:
            return 1.0
        if hasattr(pm, "polaron_B"):
            return float(pm.polaron_B())
        if hasattr(pm, "outputs") and hasattr(pm.outputs, "B"):
            return float(pm.outputs.B)
        return 1.0

    # ---- cavity scalars (unitful) ----

    @cached_property
    def cavity_lambda(self) -> QuantityLike:
        """Cavity wavelength in meters (if cavity_params exists)."""
        cp = getattr(self.qd, "cavity_params", None)
        if cp is None:
            return Q(0.0, "m")
        # your CavityParams has lambda_m property; otherwise fall back
        if hasattr(cp, "lambda_m"):
            return as_quantity(cp.lambda_m, "m")
        return as_quantity(cp.lambda_cav, "m")

    @cached_property
    def cavity_omega(self) -> QuantityLike:
        """Cavity ωc in rad/s."""
        lam = self.cavity_lambda
        if float(lam.to("m").magnitude) == 0.0:
            return Q(0.0, "rad/s")
        nu = (c / lam).to("Hz")
        return (2.0 * np.pi * nu).to("rad/s")

    @cached_property
    def cavity_kappa(self) -> QuantityLike:
        """Cavity linewidth κ = ωc / Q in rad/s."""
        cp = getattr(self.qd, "cavity_params", None)
        if cp is None:
            return Q(0.0, "rad/s")
        Qfac = float(cp.Q)
        if Qfac <= 0:
            return Q(0.0, "rad/s")
        return (self.cavity_omega / Qfac).to("rad/s")

    # ---- float views (for hot code) ----

    def omega_rad_s(self, tr: TrLike) -> float:
        return magnitude(self.omega(tr), "rad/s")

    def freq_Hz(self, tr: TrLike) -> float:
        return magnitude(self.freq(tr), "Hz")

    def wavelength_m(self, tr: TrLike) -> float:
        return magnitude(self.wavelength_vacuum(tr), "m")

    def e_plus_hv(self) -> np.ndarray:
        """Exciton '+' eigenmode Jones vector in HV basis."""
        pol = getattr(self.qd, "polarization", None)
        if pol is None:
            return np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
        return np.asarray(pol.e_plus_hv(), dtype=complex).reshape(2)

    def e_minus_hv(self) -> np.ndarray:
        """Exciton '-' eigenmode Jones vector in HV basis."""
        pol = getattr(self.qd, "polarization", None)
        if pol is None:
            return np.array([0.0 + 0j, 1.0 + 0j], dtype=complex)
        return np.asarray(pol.e_minus_hv(), dtype=complex).reshape(2)

    def overlap_to_pm(self, tr: Transition) -> tuple[float, float]:
        """
        Return (p_plus, p_minus) where p = |<e_pm | e_d(tr)>|^2.
        """
        e_d = np.asarray(self.e_pol_hv(tr), dtype=complex).reshape(2)
        e_d = e_d / np.linalg.norm(e_d)
        e_p = self.e_plus_hv()
        e_m = self.e_minus_hv()
        e_p = e_p / np.linalg.norm(e_p)
        e_m = e_m / np.linalg.norm(e_m)
        p_plus = float(abs(np.vdot(e_p, e_d)) ** 2)
        p_minus = float(abs(np.vdot(e_m, e_d)) ** 2)
        return p_plus, p_minus

    def coupled_pm_label(self, tr: Transition, *, thresh: float = 0.75) -> str:
        """
        Human label: '+', '-', 'mixed' based on overlap with exciton eigenmodes.
        """
        p_plus, p_minus = self.overlap_to_pm(tr)
        if p_plus >= thresh and p_plus > p_minus:
            return f"+ (p+={p_plus:.2f})"
        if p_minus >= thresh and p_minus > p_plus:
            return f"- (p-={p_minus:.2f})"
        return f"mixed (p+={p_plus:.2f}, p-={p_minus:.2f})"

    def _report_plain_impl(self) -> str:
        """
        Pretty, human-readable summary of derived quantities.

        Intended for debugging/logging, not for machine parsing.
        """
        lines: list[str] = []
        add = lines.append

        lines.append("========= === DerivedQD === =========---")

        # Energies
        E = self.energies
        add("Energies (eV):")
        add(f"  X1 = {magnitude(E['X1'], 'eV'):.6f} eV")
        add(f"  X2 = {magnitude(E['X2'], 'eV'):.6f} eV")
        add(f"  XX = {magnitude(E['XX'], 'eV'):.6f} eV")

        # Optional derived from EnergyStructure (if present on qd)
        es = getattr(self.qd, "energy_structure", None)
        if es is not None:
            add("Derived (eV):")
            add(
                f"  exciton_center = {
                    magnitude(es.exciton_center, 'eV'):.6f} eV"
            )
            add(f"  fss           = {magnitude(es.fss, 'eV'):.6e} eV")
            add(f"  binding       = {magnitude(es.binding, 'eV'):.6e} eV")

        pol = getattr(self.qd, "polarization", None)
        if pol is not None:
            add("Polarization (exciton eigenbasis):")
            add(f"  theta = {float(pol.theta):.4f} rad")
            add(f"  phi   = {float(pol.phi):.4f} rad")
            add(f"  Omega = {magnitude(pol.Omega, 'eV'):.6e} eV")
            ep = pol.e_plus_hv()
            em = pol.e_minus_hv()
            add(f"  e_plus(H,V)  = ({ep[0]:.3g}, {ep[1]:.3g})")
            add(f"  e_minus(H,V) = ({em[0]:.3g}, {em[1]:.3g})")
        add("Transitions (forward direction):")
        for tp in (
            TransitionPair.G_X1,
            TransitionPair.G_X2,
            TransitionPair.X1_XX,
            TransitionPair.X2_XX,
            TransitionPair.G_XX,
        ):
            # pick forward directed transition for dipole magnitude
            fwd, _ = self.t_registry.directed(tp)

            try:
                pol = getattr(self.qd, "polarization", None)
                if pol is not None:
                    p_plus, p_minus = self.overlap_to_pm(fwd)
                    add(
                        f"            couples-to: {
                            self.coupled_pm_label(fwd)}  "
                        f"(p+={p_plus:.2f}, p-={p_minus:.2f})"
                    )
            except Exception:
                pass
            nu = magnitude(self.freq(tp), "Hz")
            w = magnitude(self.omega(tp), "rad/s")
            lam_nm = magnitude(self.wavelength_vacuum(tp), "m") * 1e9

            add(
                f"  {tp.value:7s}  ν={nu:.3e} Hz   ω={
                    w:.3e} rad/s   λ={lam_nm:.2f} nm"
            )

            # dipole magnitude (if available)
            try:
                mu = magnitude(self.mu(fwd), "C*m")
                add(f"            μ({fwd.value}) = {mu:.3e} C·m")
            except Exception:
                pass

        # Phonons
        add(f"Phonons:")
        add(f"  polaron_B ⟨B⟩(t) = {float(self.polaron_B):.6f}")

        # Cavity
        add("Cavity:")
        add(f"  ωc = {magnitude(self.cavity_omega, 'rad/s'):.3e} rad/s")
        add(f"  κ  = {magnitude(self.cavity_kappa, 'rad/s'):.3e} rad/s")

        return "\n".join(lines)

    def render_report(self, *, style: dict | None = None) -> str:
        st = style or {}
        rs = ReportStyle(
            use_color=bool(st.get("use_color", True)),
            use_unicode=bool(st.get("use_unicode", True)),
            width=st.get("width", None),
        )
        if has_rich():
            return render_to_text(self, style=rs)
        return self.report()

    def print_report(self, *, style: dict | None = None) -> None:
        st = style or {}
        rs = ReportStyle(
            use_color=bool(st.get("use_color", True)),
            use_unicode=bool(st.get("use_unicode", True)),
            width=st.get("width", None),
        )
        if has_rich():
            _print_rich(self, style=rs)
        else:
            print(self.report())

    def report_plain(self) -> str:
        """
        Plain ASCII report string (safe for logs/tests).
        """
        # keep your plain report implementation here, but FIX it:
        # - no multiline f-strings
        # - no unicode characters in code
        return self._report_plain_impl()


    def report(self, *, rich: bool = True, style: dict | None = None) -> str:
        """
        Default human-facing report.
        If rich=True and rich is installed, returns a rendered rich report as text.
        Otherwise returns the plain ASCII report string.
        """
        if not rich:
            return self.report_plain()

        st = style or {}
        rs = ReportStyle(
            use_color=bool(st.get("use_color", True)),
            use_unicode=bool(st.get("use_unicode", True)),
            width=st.get("width", 120),
        )
        if has_rich():
            return render_to_text(self, style=rs)
        return self.report_plain()


