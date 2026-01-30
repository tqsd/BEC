from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from smef.core.ir.ops import OpExpr
from smef.core.ir.terms import Term, TermKind

from bec.quantum_dot.enums import Transition, TransitionPair
from bec.quantum_dot.smef.drives.emitter.coeffs import ArrayCoeff
from bec.quantum_dot.smef.drives.emitter.symbols import (
    qd_local,
    transition_symbol,
)


def build_drive_h_term(
    *,
    qd_index: int,
    drive_id: Any,
    pair: TransitionPair,
    fwd: Transition,
    bwd: Transition,
    omega_solver: np.ndarray,
    meta: Mapping[str, Any],
) -> Term:
    """
    Build the coherent drive Hamiltonian term:

      H_drive(t) = 0.5 * Omega(t) * (t_fwd + t_bwd)

    where t_* are the transition operators |dst><src| on the QD subsystem.
    """
    omega_solver = np.asarray(omega_solver, dtype=complex).reshape(-1)

    op_up = qd_local(qd_index, transition_symbol(fwd))
    op_dn = qd_local(qd_index, transition_symbol(bwd))

    op_drive = OpExpr.scale(0.5 + 0.0j, OpExpr.summation((op_up, op_dn)))

    return Term(
        kind=TermKind.H,
        op=op_drive,
        coeff=ArrayCoeff(omega_solver),
        label="H_drive_%s_%s" % (str(drive_id), str(pair.value)),
        meta=dict(meta),
    )
