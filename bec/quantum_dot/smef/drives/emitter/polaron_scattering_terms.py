from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from smef.core.ir.ops import OpExpr
from smef.core.ir.terms import Term, TermKind

from bec.quantum_dot.enums import QDState, Transition, TransitionPair
from bec.quantum_dot.models.phonon_model import PolaronDriveRates
from bec.quantum_dot.smef.drives.emitter.coeffs import ArrayCoeff
from bec.quantum_dot.smef.drives.emitter.symbols import (
    proj_symbol,
    qd_local,
    transition_symbol,
)


def _tr_from_states(src: QDState, dst: QDState) -> Transition:
    """
    Convert (src, dst) into Transition enum value Transition.SRC_DST.

    This matches your Transition encoding and qd4_transition_op convention:
      Transition.SRC_DST -> operator |DST><SRC|
    """
    key = "%s_%s" % (src.value, dst.value)
    return Transition(key)


def build_polaron_scattering_c_terms(
    *,
    qd_index: int,
    drive_id: Any,
    pair: TransitionPair,
    dst_state: QDState,
    src_state: QDState,
    omega_solver: np.ndarray,
    detuning_rad_s: np.ndarray,
    time_unit_s: float,
    polaron_rates: PolaronDriveRates | None,
    scale: float,
    meta: Mapping[str, Any],
    b_polaron: float = 1.0,
    Nt: int = 4096,
) -> list[Term]:
    """
    Polaron ME scattering terms (time-dependent), emitted per driven TransitionPair.

    Returns three collapse terms:
      - down: L_down(t) = sqrt(g_down_solver(t)) * |dst><src|
      - up:   L_up(t)   = sqrt(g_up_solver(t))   * |src><dst|
      - cd:   L_cd(t)   = sqrt(g_cd_solver(t))   * (P_dst - P_src)

    Unit conventions
    ---------------
    - polaron_rates.gamma_dressed_rates_1_s returns physical rates in 1/s.
    - Convert to solver rates via gamma_solver = gamma_phys * time_unit_s.
    - Collapse term uses coeff = sqrt(gamma_solver(t)).

    Notes
    -----
    - Because your DriveStrengthModel already applies polaron renormalization
      omega_solver *= B, you must not apply B again here. Default b_polaron=1.0.
    """
    if polaron_rates is None or (not bool(polaron_rates.enabled)):
        return []

    sc = float(scale)
    if sc <= 0.0:
        return []

    s = float(time_unit_s)
    if s <= 0.0:
        raise ValueError("time_unit_s must be > 0")

    omega_solver = np.asarray(omega_solver, dtype=complex).reshape(-1)
    detuning_rad_s = np.asarray(detuning_rad_s, dtype=float).reshape(-1)
    if omega_solver.size != detuning_rad_s.size:
        raise ValueError("detuning_rad_s must have same length as omega_solver")

    # Physical rates (1/s)
    g_down_1_s, g_up_1_s, g_cd_1_s = polaron_rates.gamma_dressed_rates_1_s(
        omega_solver=omega_solver,
        detuning_rad_s=detuning_rad_s,
        time_unit_s=s,
        b_polaron=float(b_polaron),
        Nt=int(Nt),
    )

    g_down_1_s = np.asarray(g_down_1_s, dtype=float).reshape(-1)
    g_up_1_s = np.asarray(g_up_1_s, dtype=float).reshape(-1)
    g_cd_1_s = np.asarray(g_cd_1_s, dtype=float).reshape(-1)

    # Apply external scale as a calibration factor on the *rates*
    g_down_1_s = sc * g_down_1_s
    g_up_1_s = sc * g_up_1_s
    g_cd_1_s = sc * g_cd_1_s

    # Convert to solver rates
    g_down_solver = g_down_1_s * s
    g_up_solver = g_up_1_s * s
    g_cd_solver = g_cd_1_s * s

    sqrt_down = np.sqrt(np.maximum(g_down_solver, 0.0)).astype(complex)
    sqrt_up = np.sqrt(np.maximum(g_up_solver, 0.0)).astype(complex)
    sqrt_cd = np.sqrt(np.maximum(g_cd_solver, 0.0)).astype(complex)

    # Build fixed operators using your symbol helpers
    tr_down = _tr_from_states(src_state, dst_state)  # SRC_DST
    tr_up = _tr_from_states(dst_state, src_state)  # DST_SRC

    op_down = qd_local(qd_index, transition_symbol(tr_down))
    op_up = qd_local(qd_index, transition_symbol(tr_up))

    P_dst = qd_local(qd_index, proj_symbol(dst_state))
    P_src = qd_local(qd_index, proj_symbol(src_state))
    P_diff = OpExpr.summation([P_dst, OpExpr.scale(complex(-1.0), P_src)])

    out: list[Term] = []

    out.append(
        Term(
            kind=TermKind.C,
            op=op_down,
            coeff=ArrayCoeff(sqrt_down),
            label="L_pol_scatt_down_%s_%s" % (str(drive_id), str(pair.value)),
            meta={
                **dict(meta),
                "kind": "polaron_scatt",
                "chan": "down",
                "pair": pair.value,
                "src": src_state.value,
                "dst": dst_state.value,
                "scale": float(sc),
                "Nt": int(Nt),
                "b_polaron": float(b_polaron),
            },
        )
    )

    out.append(
        Term(
            kind=TermKind.C,
            op=op_up,
            coeff=ArrayCoeff(sqrt_up),
            label="L_pol_scatt_up_%s_%s" % (str(drive_id), str(pair.value)),
            meta={
                **dict(meta),
                "kind": "polaron_scatt",
                "chan": "up",
                "pair": pair.value,
                "src": dst_state.value,
                "dst": src_state.value,
                "scale": float(sc),
                "Nt": int(Nt),
                "b_polaron": float(b_polaron),
            },
        )
    )

    out.append(
        Term(
            kind=TermKind.C,
            op=P_diff,
            coeff=ArrayCoeff(sqrt_cd),
            label="L_pol_scatt_cd_%s_%s" % (str(drive_id), str(pair.value)),
            meta={
                **dict(meta),
                "kind": "polaron_scatt",
                "chan": "cd",
                "pair": pair.value,
                "src": src_state.value,
                "dst": dst_state.value,
                "scale": float(sc),
                "Nt": int(Nt),
                "b_polaron": float(b_polaron),
            },
        )
    )

    return out
