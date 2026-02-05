from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from smef.core.units import QuantityLike, magnitude

# If you have these enums available, great; otherwise the code still works
# because it only uses .value / str fallbacks.
try:
    from bec.quantum_dot.enums import QDState, Transition, TransitionPair
except Exception:  # pragma: no cover
    QDState = object  # type: ignore
    Transition = object  # type: ignore
    TransitionPair = object  # type: ignore


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _key_str(k: Any) -> str:
    return getattr(k, "value", None) or _safe_str(k)


def _fmt_qty(q: Any, unit: str, *, precision: int = 6) -> str:
    """
    Format a unitful QuantityLike into `value unit` (best-effort).
    """
    try:
        val = float(magnitude(q, unit))
        return f"{val:.{precision}g} {unit}"
    except Exception:
        return _safe_str(q)


def _fmt_complex(z: complex, *, precision: int = 6) -> str:
    return f"({z.real:.{precision}g}{z.imag:+.{precision}g}j)"


def _indent(lines: Iterable[str], n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln else ln for ln in lines)


def _section(title: str, body: str) -> str:
    if not body.strip():
        return ""
    bar = "-" * len(title)
    return f"{title}\n{bar}\n{body}\n"


@dataclass(frozen=True)
class PlainReportMixin:
    """
    Plaintext reporting for DerivedQD.

    Assumes the host object has:
      - self.qd (QuantumDot)
      - mixins providing: energies, transition_energy, omega, freq, mu, e_pol_hv,
        drive_projection, rates, phonon_outputs, polaron_B, gamma_phi_eid_scale
      - optionally: exciton_effective_polarization_report()
    """

    # -------- top-level entrypoint --------

    def report(self) -> str:
        parts: list[str] = []

        parts.append(self._header_block())
        parts.append(self._params_block())
        parts.append(self._energies_block())
        parts.append(self._transitions_block())
        parts.append(self._dipoles_block())
        parts.append(self._rates_block())
        parts.append(self._phonons_block())
        parts.append(self._exciton_basis_block())

        out = "\n".join(p for p in parts if p.strip())
        return out.strip() + "\n"

    # -------- blocks --------

    def _header_block(self) -> str:
        qd = getattr(self, "qd", None)
        name = (
            qd.__class__.__name__ if qd is not None else self.__class__.__name__
        )

        modes = None
        try:
            cb = qd.compile_bundle(units=None)  # type: ignore[arg-type]
            modes = getattr(cb, "modes", None)
        except Exception:
            modes = None

        lines = [f"Quantum dot report: {name}"]

        if modes is not None and hasattr(modes, "dims"):
            try:
                dims = list(modes.dims())
                lines.append(f"modes.dims: {dims}  (D={int(np.prod(dims))})")
            except Exception:
                pass

        # Basic presence flags
        try:
            has_cavity = getattr(qd, "cavity", None) is not None
            has_phonons = getattr(qd, "phonons", None) is not None
            lines.append(f"cavity: {'yes' if has_cavity else 'no'}")
            lines.append(f"phonons: {'yes' if has_phonons else 'no'}")
        except Exception:
            pass

        return "\n".join(lines) + "\n"

    def _params_block(self) -> str:
        qd = getattr(self, "qd", None)
        if qd is None:
            return ""

        lines: list[str] = []

        energy = getattr(qd, "energy", None)
        dipoles = getattr(qd, "dipoles", None)
        cavity = getattr(qd, "cavity", None)
        phonons = getattr(qd, "phonons", None)

        # Just show dataclass-ish reprs; keeps it simple and robust.
        lines.append("energy:")
        lines.append(_indent([_safe_str(energy)]))
        lines.append("dipoles:")
        lines.append(_indent([_safe_str(dipoles)]))

        if cavity is not None:
            lines.append("cavity:")
            lines.append(_indent([_safe_str(cavity)]))

        if phonons is not None:
            lines.append("phonons:")
            lines.append(_indent([_safe_str(phonons)]))

        return _section("Parameters", "\n".join(lines))

    def _energies_block(self) -> str:
        if not hasattr(self, "energies"):
            return ""

        lines: list[str] = []

        # Absolute energies
        try:
            E: Mapping[Any, QuantityLike] = getattr(self, "energies")
            lines.append("state energies (eV):")
            st_items = list(E.items())
            for st, val in st_items:
                lines.append(
                    f"  {_key_str(st):>4s}: {_fmt_qty(val, 'eV', precision=8)}"
                )
        except Exception:
            pass

        # Transition energies / frequencies
        try:
            TE: Mapping[Any, QuantityLike] = getattr(self, "transition_energy")
            lines.append("")
            lines.append("transition energies (eV) and frequencies:")
            for tr, dE in TE.items():
                tr_name = _key_str(tr)
                dE_s = _fmt_qty(dE, "eV", precision=8)
                try:
                    w = getattr(self, "omega")(tr)
                    f = getattr(self, "freq")(tr)
                    w_s = _fmt_qty(w, "rad/s", precision=6)
                    f_s = _fmt_qty(f, "Hz", precision=6)
                    lines.append(
                        f"  {tr_name:>8s}: dE={dE_s}  omega={w_s}  f={f_s}"
                    )
                except Exception:
                    lines.append(f"  {tr_name:>8s}: dE={dE_s}")
        except Exception:
            pass

        return _section("Energies", "\n".join(lines))

    def _transitions_block(self) -> str:
        # Uses your transition registry if available
        tr_reg = getattr(self, "t_registry", None)
        if tr_reg is None:
            return ""

        lines: list[str] = []

        # Show directed and pairing relations if the registry supports it
        # (I’m not assuming method names beyond what you’ve already used: directed(pair)).
        try:
            lines.append("transition registry:")
            lines.append(_indent([f"type: {tr_reg.__class__.__name__}"]))
        except Exception:
            pass

        # If you have an enum list, show it. Otherwise skip.
        try:
            # Common pattern: Transition enum is iterable
            if hasattr(Transition, "__iter__"):
                trs = list(Transition)  # type: ignore[arg-type]
                lines.append("directed transitions:")
                for tr in trs:
                    lines.append(f"  - {_key_str(tr)}")
        except Exception:
            pass

        # If TransitionPair exists, show mapping to directed
        try:
            if hasattr(TransitionPair, "__iter__") and hasattr(
                tr_reg, "directed"
            ):
                pairs = list(TransitionPair)  # type: ignore[arg-type]
                lines.append("")
                lines.append("pair -> (forward, backward):")
                for p in pairs:
                    try:
                        fwd, bwd = tr_reg.directed(p)
                        lines.append(
                            f"  {_key_str(p):>8s}: ({_key_str(fwd)}, {
                                _key_str(bwd)})"
                        )
                    except Exception:
                        lines.append(f"  {_key_str(p):>8s}: <unavailable>")
        except Exception:
            pass

        return _section("Transitions", "\n".join(lines))

    def _dipoles_block(self) -> str:
        if not hasattr(self, "mu") and not hasattr(self, "e_pol_hv"):
            return ""

        lines: list[str] = []

        # For each directed transition: mu and polarization vector in HV
        trs: Sequence[Any] = []
        try:
            if hasattr(Transition, "__iter__"):
                trs = list(Transition)  # type: ignore[arg-type]
        except Exception:
            trs = []

        if trs:
            lines.append("dipoles per transition:")
            for tr in trs:
                row: list[str] = []
                row.append(f"{_key_str(tr):>8s}")

                try:
                    mu = getattr(self, "mu")(tr)
                    row.append(f"mu={_fmt_qty(mu, 'C*m', precision=6)}")
                except Exception:
                    pass

                try:
                    v = getattr(self, "e_pol_hv")(tr)
                    v = np.asarray(v, dtype=complex).reshape(2)
                    row.append(
                        f"e_pol_hv=[{_fmt_complex(complex(v[0]))}, {
                            _fmt_complex(complex(v[1]))}]"
                    )
                except Exception:
                    pass

                lines.append("  " + "  ".join(row))
        else:
            # Fallback: show whatever dipole params repr is
            try:
                dp = getattr(getattr(self, "qd", None), "dipoles", None)
                lines.append(_safe_str(dp))
            except Exception:
                pass

        return _section("Dipoles", "\n".join(lines))

    def _rates_block(self) -> str:
        if not hasattr(self, "rates"):
            return ""

        lines: list[str] = []
        try:
            r: Mapping[str, QuantityLike] = getattr(self, "rates")
            if not r:
                return _section("Rates", "no rates available")
            lines.append("rates (1/s):")
            for k in sorted(r.keys()):
                lines.append(
                    f"  {k:>24s}: {_fmt_qty(r[k], '1/s', precision=6)}"
                )
        except Exception:
            return _section("Rates", "failed to compute rates")
        return _section("Rates", "\n".join(lines))

    def _phonons_block(self) -> str:
        # Only report if phonons are configured or outputs exist
        po = getattr(self, "phonon_outputs", None)
        qd = getattr(self, "qd", None)
        has_params = False
        try:
            has_params = (qd is not None) and (
                getattr(qd, "phonons", None) is not None
            )
        except Exception:
            has_params = False

        if (po is None) and (not has_params):
            return ""

        lines: list[str] = []

        # Summary scalars
        try:
            scale = getattr(self, "gamma_phi_eid_scale", 0.0)
            lines.append(f"gamma_phi_eid_scale: {float(scale):.6g}")
        except Exception:
            pass

        # Polaron B per transition if possible
        try:
            Bmap = getattr(po, "B_polaron_per_transition", None)
            if isinstance(Bmap, Mapping) and Bmap:
                lines.append("polaron B per transition:")
                for tr, B in Bmap.items():
                    lines.append(f"  {_key_str(tr):>8s}: {float(B):.6g}")
        except Exception:
            pass

        # Any phonon-provided rates
        try:
            pr = getattr(po, "rates", None)
            if isinstance(pr, Mapping) and pr:
                lines.append("phonon rates merged (1/s):")
                for k, v in pr.items():
                    lines.append(
                        f"  {_key_str(k):>24s}: {
                            _fmt_qty(v, '1/s', precision=6)}"
                    )
        except Exception:
            pass

        if not lines:
            lines.append("phonons configured, but no outputs to report")

        return _section("Phonons", "\n".join(lines))

    def _exciton_basis_block(self) -> str:
        if not hasattr(self, "exciton_effective_polarization_report"):
            return ""

        try:
            rep = getattr(self, "exciton_effective_polarization_report")()
        except Exception:
            return _section("Exciton basis", "exciton basis report unavailable")

        lines: list[str] = []
        try:
            lines.append(f"Delta_eV: {float(rep.get('Delta_eV', 0.0)):.6g}")
            lines.append(
                f"delta_prime_eV: {float(rep.get('delta_prime_eV', 0.0)):.6g}"
            )
            lines.append(f"theta_rad: {float(rep.get('theta_rad', 0.0)):.6g}")
            lines.append(f"theta_deg: {float(rep.get('theta_deg', 0.0)):.6g}")

            eig = rep.get("eigen_relative_eV", None)
            if isinstance(eig, (tuple, list)) and len(eig) == 2:
                lines.append(
                    f"eigen relative eV: (+{float(eig[0]):.6g}, {float(eig[1]):.6g})"
                )

            eff = rep.get("effective", {})
            if isinstance(eff, Mapping) and eff:
                lines.append(
                    "effective dipole polarization (HV power fractions):"
                )
                for name in sorted(eff.keys()):
                    item = eff[name]
                    pH = float(item.get("pH", 0.0))
                    pV = float(item.get("pV", 0.0))
                    hv = item.get("hv_vec", None)
                    if isinstance(hv, (tuple, list)) and len(hv) == 2:
                        z0 = complex(hv[0])
                        z1 = complex(hv[1])
                        lines.append(
                            f"  {name}: pH={pH:.6g} pV={pV:.6g}  hv=[{
                                _fmt_complex(z0)}, {_fmt_complex(z1)}]"
                        )
                    else:
                        lines.append(f"  {name}: pH={pH:.6g} pV={pV:.6g}")
        except Exception:
            return _section("Exciton basis", "failed to format exciton report")

        return _section("Exciton basis", "\n".join(lines))
