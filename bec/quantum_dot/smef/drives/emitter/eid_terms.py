from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from smef.core.ir.terms import Term, TermKind

from bec.quantum_dot.enums import TransitionPair
from bec.quantum_dot.models.phonon_model import PolaronDriveRates
from bec.quantum_dot.smef.drives.emitter.coeffs import ArrayCoeff
from bec.quantum_dot.smef.drives.emitter.symbols import proj_symbol, qd_local


def _abs2(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=complex).reshape(-1)
    re = np.asarray(np.real(z), dtype=float)
    im = np.asarray(np.imag(z), dtype=float)
    return (re * re) + (im * im)


def build_eid_c_term_phenom(
    *,
    qd_index: int,
    drive_id: Any,
    pair: TransitionPair,
    dst_proj_state,
    omega_solver: np.ndarray,
    eid_scale: float,
    meta: Mapping[str, Any],
) -> Optional[Term]:
    """
    Phenomenological EID:

      gamma_solver(t) = eid_scale * |Omega_solver(t)|^2
      L_eid(t) = sqrt(gamma_solver(t)) * P_high

    Returns a collapse term or None if eid_scale <= 0.
    """
    scale = float(eid_scale)
    if scale <= 0.0:
        return None

    omega_solver = np.asarray(omega_solver, dtype=complex).reshape(-1)
    gamma_solver = scale * _abs2(omega_solver)
    sqrt_gamma = np.sqrt(np.maximum(gamma_solver, 0.0)).astype(complex)

    P_high = qd_local(qd_index, proj_symbol(dst_proj_state))

    return Term(
        kind=TermKind.C,
        op=P_high,
        coeff=ArrayCoeff(sqrt_gamma),
        label="L_eid_%s_%s" % (str(drive_id), str(pair.value)),
        meta={
            **dict(meta),
            "kind": "phonon_eid",
            "model": "phenomenological",
            "scale": float(scale),
        },
    )


def build_eid_c_term_polaron(
    *,
    qd_index: int,
    drive_id: Any,
    pair: TransitionPair,
    dst_proj_state: Any,
    omega_solver: np.ndarray,
    detuning_rad_s: np.ndarray,
    time_unit_s: float,
    polaron_rates: Optional[PolaronDriveRates],
    scale: float,
    meta: Mapping[str, Any],
) -> Optional[Term]:
    """
    Polaron-shaped EID (drive-dependent):

      gamma_1_s(t) = polaron_rates.gamma_eid_1_s(omega_solver, detuning_rad_s, time_unit_s, scale)
      gamma_solver(t) = gamma_1_s(t) * time_unit_s
      L_eid(t) = sqrt(gamma_solver(t)) * P_high

    Returns None if polaron_rates is None or disabled or scale <= 0.
    """
    if polaron_rates is None or not bool(polaron_rates.enabled):
        return None

    sc = float(scale)
    if sc <= 0.0:
        return None

    s = float(time_unit_s)
    if s <= 0.0:
        raise ValueError("time_unit_s must be > 0")

    omega_solver = np.asarray(omega_solver, dtype=complex).reshape(-1)
    detuning_rad_s = np.asarray(detuning_rad_s, dtype=float).reshape(-1)
    if detuning_rad_s.size != omega_solver.size:
        raise ValueError("detuning_rad_s must have same length as omega_solver")

    gamma_1_s = polaron_rates.gamma_eid_1_s(
        omega_solver=omega_solver,
        detuning_rad_s=detuning_rad_s,
        time_unit_s=s,
        scale=sc,
    )
    gamma_1_s = np.asarray(gamma_1_s, dtype=float).reshape(-1)

    gamma_solver = gamma_1_s * s
    sqrt_gamma = np.sqrt(np.maximum(gamma_solver, 0.0)).astype(complex)

    P_high = qd_local(qd_index, proj_symbol(dst_proj_state))

    return Term(
        kind=TermKind.C,
        op=P_high,
        coeff=ArrayCoeff(sqrt_gamma),
        label="L_eid_polaron_%s_%s" % (str(drive_id), str(pair.value)),
        meta={
            **dict(meta),
            "kind": "phonon_eid",
            "model": "polaron_minimal",
            "scale": float(sc),
        },
    )
