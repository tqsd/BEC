from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.constants import epsilon_0 as _eps0, c as _c, hbar as _hbar, e as _e

from bec.params.transitions import Transition
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.models.phonon_model import PhononModel
from bec.simulation.protocols import RatesStage
from bec.simulation.types import DriveCoefficients, RatesBundle, ResolvedDrive
from bec.quantum_dot.me.coeffs import CoeffExpr, as_coeff


def _as_float(x, name: str) -> float:
    try:
        return float(x)
    except Exception as exc:
        raise TypeError(f"{name} must be float-like, got {type(x)!r}") from exc


def _ev_to_omega(e_ev: float) -> float:
    return float(e_ev) * _e / _hbar


def _transition_omega_rad_s(qd: QuantumDot, tr: Transition) -> float:
    el = qd.energy_levels
    if tr == Transition.G_X1:
        return _ev_to_omega(el.X1)
    if tr == Transition.G_X2:
        return _ev_to_omega(el.X2)
    if tr == Transition.X1_XX:
        return _ev_to_omega(el.XX - el.X1)
    if tr == Transition.X2_XX:
        return _ev_to_omega(el.XX - el.X2)
    raise ValueError(f"Unsupported transition for radiative omega: {tr!r}")


def _purcell_factor_from_cavity(qd: QuantumDot) -> float:
    cav = getattr(qd, "cavity_params", None)
    if cav is None:
        return 1.0

    Q = _as_float(getattr(cav, "Q", 0.0), "cavity_params.Q")
    Veff_um3 = _as_float(
        getattr(cav, "Veff_um3", 0.0), "cavity_params.Veff_um3"
    )
    lam_nm = _as_float(
        getattr(cav, "lambda_nm", 0.0), "cavity_params.lambda_nm"
    )
    n = _as_float(getattr(cav, "n", 3.5), "cavity_params.n")

    if Q <= 0.0 or Veff_um3 <= 0.0 or lam_nm <= 0.0 or n <= 0.0:
        return 1.0

    lam_m = lam_nm * 1.0e-9
    Veff_m3 = Veff_um3 * 1.0e-18

    fp = (3.0 / (4.0 * np.pi * np.pi)) * ((lam_m / n) ** 3) * (Q / Veff_m3)
    if not np.isfinite(fp) or fp <= 0.0:
        return 1.0
    return float(fp)


def _dipole_magnitude_Cm(qd: QuantumDot, tr: Transition) -> Optional[float]:
    dp = getattr(qd, "dipole_params", None)
    if dp is None:
        return None

    # --- NEW: common "value fields" ---
    for attr in ("dipole_moment_Cm", "d_Cm", "mu_Cm_value", "d_mag_Cm_value"):
        v = getattr(dp, attr, None)
        if v is not None and not callable(v):
            try:
                vv = float(v)
                if vv > 0:
                    return abs(vv)
            except Exception:
                pass

    # --- existing callable hooks ---
    fn = getattr(dp, "d_mag_Cm", None)
    if callable(fn):
        return float(abs(fn(tr)))

    fn = getattr(dp, "mu_Cm", None)
    if callable(fn):
        return float(abs(fn(tr)))

    fn = getattr(dp, "d_vec_hv", None)
    if callable(fn):
        v = np.asarray(fn(tr), dtype=complex).reshape(2)
        return float(np.linalg.norm(v))

    return None


def _gamma_spontaneous_1_s(omega_rad_s: float, dipole_Cm: float) -> float:
    w = float(omega_rad_s)
    d = float(dipole_Cm)
    if w <= 0.0 or d <= 0.0:
        return 0.0
    g0 = (w**3) * (d**2) / (3.0 * np.pi * _eps0 * _hbar * (_c**3))
    if not np.isfinite(g0) or g0 < 0.0:
        return 0.0
    return float(g0)


def _const_coeff(val_solver: float) -> CoeffExpr:
    v = float(val_solver)

    def f(t: float, args=None, *, _v=v) -> float:
        return _v

    return as_coeff(f)


def _get_rd_kind(rd: ResolvedDrive) -> str:
    return str(getattr(rd, "kind", "1ph"))


@dataclass(frozen=True)
class DefaultRatesStage(RatesStage):
    """
    Drive-aware rates evaluation.

    Returns RatesBundle.rates in solver units:
      - radiative cascade: constants from QD specs (+ Purcell)
      - phonon pure dephasing: constants from PhononModel.compute()
      - drive-dependent EID: phenomenological function of Omega(t)
        controlled by PhononParams.gamma_phi_eid_scale

    Also returns meta:
      - B_polaron (polaron dressing factor)
      - purcell_factor
      - eid_model info
    """

    fallback_gamma0_1_s: float = 1.0e9
    use_purcell: bool = True

    eid_power: float = 1.0
    eid_floor_1_s: float = 0.0

    def compute(
        self,
        *,
        qd: QuantumDot,
        resolved: Tuple[ResolvedDrive, ...],
        drive_coeffs: Optional[Dict[str, DriveCoefficients]] = None,
        time_unit_s: float,
        tlist: np.ndarray,
    ) -> RatesBundle:
        """
        NOTE: this stage is drive-aware, so pass drive_coeffs from DriveStrengthModel.
        If you do not pass drive_coeffs, EID is disabled (rates will be drive-agnostic).
        """
        s = _as_float(time_unit_s, "time_unit_s")
        tlist = np.asarray(tlist, dtype=float)

        rates: Dict[str, CoeffExpr] = {}
        meta: Dict[str, object] = {}

        # -------------------------
        # Radiative cascade constants (phys -> solver)
        # -------------------------
        fp = _purcell_factor_from_cavity(qd) if self.use_purcell else 1.0
        meta["purcell_factor"] = float(fp)

        for tr, key in [
            (Transition.G_X1, "gamma_x1_g"),
            (Transition.G_X2, "gamma_x2_g"),
            (Transition.X1_XX, "gamma_xx_x1"),
            (Transition.X2_XX, "gamma_xx_x2"),
        ]:
            omega = _transition_omega_rad_s(qd, tr)
            dmag = _dipole_magnitude_Cm(qd, tr)

            if dmag is None or dmag <= 0.0:
                gamma_phys = float(self.fallback_gamma0_1_s)
            else:
                gamma_phys = _gamma_spontaneous_1_s(omega, dmag) * (
                    1 + float(fp)
                )

            rates[key] = _const_coeff(gamma_phys * s)

            print(tr, "omega", omega, "dmag", dmag, "gamma_phys", gamma_phys)

        # -------------------------
        # Phonon constants + B_polaron
        # -------------------------
        pp = getattr(qd, "phonon_params", None)
        Bmap: Dict[Transition, float] = {}
        eid_scale = 0.0

        if pp is not None:
            pm = PhononModel(qd.energy_levels, pp)

            out = pm.compute()
            Bmap = dict(out.B_polaron_per_transition or {})
            meta["B_polaron_per_transition"] = {
                str(k): float(v) for k, v in Bmap.items()
            }
            meta["phonon_model"] = str(getattr(pp, "model", "unknown"))

            for name, gamma_phys in out.rates_1_s.items():
                rates[name] = _const_coeff(float(gamma_phys) * s)

            eid_scale = float(getattr(pp, "gamma_phi_eid_scale", 0.0))

        # -------------------------
        # Drive-dependent EID (phenomenological)
        # -------------------------
        if eid_scale > 0.0 and drive_coeffs is not None:
            meta["eid_enabled"] = True
            meta["eid_scale"] = float(eid_scale)
            meta["eid_power"] = float(self.eid_power)
            meta["eid_floor_1_s"] = float(self.eid_floor_1_s)

            # One EID rate for the exciton manifold (apply where you want in CollapseComposer)
            rates["gamma_eid_X"] = as_coeff(
                self._make_eid_coeff(
                    qd=qd,
                    resolved=resolved,
                    drive_coeffs=drive_coeffs,
                    time_unit_s=s,
                    eid_scale=eid_scale,
                    B_polaron_map=Bmap,
                )
            )
        else:
            meta["eid_enabled"] = False

        return RatesBundle(rates=rates, args={}, meta=meta)

    def _make_eid_coeff(
        self,
        *,
        qd: QuantumDot,
        resolved: Tuple[ResolvedDrive, ...],
        drive_coeffs: Dict[str, DriveCoefficients],
        time_unit_s: float,
        eid_scale: float,
        B_polaron_map: Dict[Transition, float],
    ):
        s = float(time_unit_s)
        p = float(self.eid_power)
        floor = float(self.eid_floor_1_s)
        scale = float(eid_scale)

        # Per-transition polaron dressing factors; default to 1 if missing
        Bmap: Dict[Transition, float] = dict(B_polaron_map or {})

        def B_1ph(tr: Transition) -> float:
            return float(Bmap.get(tr, 1.0))

        def B_2ph() -> float:
            return float(Bmap.get(Transition.G_XX, 1.0))

        def coeff(t_solver: float, args=None) -> float:
            t = float(t_solver)

            # Track the maximum *effective* coherent coupling (after polaron renorm)
            omega_eff_max = 0.0

            for rd in resolved:
                dc = drive_coeffs.get(rd.drive_id)
                if dc is None:
                    continue

                kind = _get_rd_kind(rd)

                if kind == "1ph":
                    # dc.omega_by_transition: Dict[Transition, CoeffExpr]
                    for tr, om in dc.omega_by_transition.items():
                        # Convert solver coefficient to physical rad/s
                        om_phys = abs(complex(om(t, args))) / s

                        # Apply transition-specific polaron factor
                        omega_eff = B_1ph(tr) * float(om_phys)

                        if omega_eff > omega_eff_max:
                            omega_eff_max = omega_eff

                elif kind == "2ph":
                    if dc.omega_2ph is not None:
                        om2_phys = abs(complex(dc.omega_2ph(t, args))) / s

                        # Apply 2ph polaron factor (G<->XX)
                        omega_eff = B_2ph() * float(om2_phys)

                        if omega_eff > omega_eff_max:
                            omega_eff_max = omega_eff

            # Phenomenological EID rate in physical 1/s
            gamma_phys = floor + scale * (omega_eff_max**p)

            # Return in solver units
            return float(gamma_phys) * s

        return coeff
