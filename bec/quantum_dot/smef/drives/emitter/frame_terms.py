from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from smef.core.ir.terms import Term, TermKind

from bec.quantum_dot.enums import QDState
from bec.quantum_dot.smef.drives.emitter.coeffs import ArrayCoeff
from bec.quantum_dot.smef.drives.emitter.frame_solver import eps_to_solver_coeff
from bec.quantum_dot.smef.drives.emitter.symbols import proj_symbol, qd_local


def build_frame_h_terms(
    *,
    qd_index: int,
    eps_rad_s_by_state: Mapping[QDState, np.ndarray],
    time_unit_s: float,
    label_prefix: str = "H_frame",
    meta: Mapping[str, Any] | None = None,
) -> tuple[Term, ...]:
    """
    Emit one Hamiltonian Term per state:
        H_frame = sum_s eps_s(t) * |s><s|

    With current SMEF IR, this is the correct representation because each
    summand needs its own time-dependent coefficient.
    """
    meta_d = dict(meta or {})
    out: list[Term] = []

    for st, eps_rad_s in eps_rad_s_by_state.items():
        coeff_solver = eps_to_solver_coeff(eps_rad_s, time_unit_s=time_unit_s)
        P = qd_local(int(qd_index), proj_symbol(st))

        out.append(
            Term(
                kind=TermKind.H,
                op=P,
                coeff=ArrayCoeff(coeff_solver),
                label="%s_%s" % (str(label_prefix), str(st.value)),
                meta=dict(meta_d),
            )
        )

    return tuple(out)
