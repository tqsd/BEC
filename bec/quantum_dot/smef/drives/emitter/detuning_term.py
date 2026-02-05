from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from smef.core.ir.ops import OpExpr
from smef.core.ir.terms import Term, TermKind

from bec.quantum_dot.enums import QDState, TransitionPair
from bec.quantum_dot.smef.drives.emitter.coeffs import ArrayCoeff
from bec.quantum_dot.smef.drives.emitter.symbols import proj_symbol, qd_local


def build_detuning_h_term(
    *,
    qd_index: int,
    drive_id: Any,
    pair: TransitionPair,
    src: QDState,
    dst: QDState,
    detuning_rad_s: np.ndarray,
    time_unit_s: float,
    meta: Mapping[str, Any],
) -> Term:
    """
    Build detuning Hamiltonian term:

      H_det(t) = -(0.5 * Delta(t)) * (P_high - P_low)

    IR coefficient must be in solver units:
      coeff_solver(t) = -(0.5 * Delta_rad_s(t)) * time_unit_s
    """
    detuning_rad_s = np.asarray(detuning_rad_s, dtype=float).reshape(-1)
    s = float(time_unit_s)

    coeff_solver = (-(0.5 * detuning_rad_s) * s).astype(complex)

    P_high = qd_local(qd_index, proj_symbol(dst))
    P_low = qd_local(qd_index, proj_symbol(src))

    op_det = OpExpr.summation((P_high, OpExpr.scale(-1.0 + 0.0j, P_low)))

    return Term(
        kind=TermKind.H,
        op=op_det,
        coeff=ArrayCoeff(coeff_solver),
        label="H_det_%s_%s" % (str(drive_id), str(pair.value)),
        meta=dict(meta),
    )
