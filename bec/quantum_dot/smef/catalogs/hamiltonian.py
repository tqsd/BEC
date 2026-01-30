from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from smef.core.ir.ops import OpExpr
from smef.core.ir.terms import Term, TermKind
from smef.core.model.protocols import TermCatalogProto
from smef.core.units import Q, hbar, magnitude

from bec.quantum_dot.enums import Transition
from bec.quantum_dot.smef.catalogs.base import FrozenCatalog
from bec.quantum_dot.smef.modes import QDModes, QDModeKey
from .base import _delta_prime_eV, _proj, _t


@dataclass(frozen=True)
class QDHamiltonianCatalog(FrozenCatalog):
    r"""
    Static Hamiltonian terms for the 4-level quantum dot in SMEF IR.

    This catalog intentionally emits only *system* (drive-independent) terms.
    Drive-dependent Hamiltonian contributions must be produced by the drive
    pipeline (decoder/strength/emitter).

    Emitted term(s)
    ---------------
    Exciton subspace FSS + mixing in the {X1, X2} basis, in a rotating/RWA frame
    where only slow terms are kept:

    .. math::

        H_{X} =
        \frac{\Delta}{2}\left(|X_1\rangle\langle X_1| - |X_2\rangle\langle X_2|\right)
        + \delta' \left(|X_1\rangle\langle X_2| + |X_2\rangle\langle X_1|\right)

    where:

    - :math:`\Delta = E_{X1} - E_{X2}` comes from ``qd.energy`` (unitful eV).
    - :math:`\delta'` comes from ``qd.mixing.delta_prime`` (unitful eV).

    Unit boundary
    -------------
    Energies in eV are converted to angular frequencies (rad/s) and then to solver
    units by multiplying with ``units.time_unit_s``. The IR stores dimensionless
    solver-coefficients.

    Operator conventions
    --------------------
    - ``proj_X1`` and ``proj_X2`` are projectors in the QD subsystem.
    - Transition symbols follow: ``t_SRC_DST = |DST><SRC|``.
      We form off-diagonal exciton operators via products:
      ``|X1><X2| = t_G_X1 * t_X2_G`` and ``|X2><X1| = t_G_X2 * t_X1_G``.
    """

    @classmethod
    def from_qd(
        cls,
        qd,
        *,
        units,
        modes: Optional[QDModes] = None,
    ) -> TermCatalogProto:
        if units is None:
            raise ValueError("units must be provided")

        qd_i = 0 if modes is None else int(modes.index_of(QDModeKey.qd()))
        time_unit_s = float(units.time_unit_s)

        # Delta = E(X1) - E(X2) (eV)
        e_x1 = float(magnitude(qd.energy.X1, "eV"))
        e_x2 = float(magnitude(qd.energy.X2, "eV"))
        delta_eV = float(e_x1 - e_x2)
        # delta_prime (eV)
        dp_eV = float(_delta_prime_eV(qd))

        # If both are zero, emit nothing (clean IR)
        if abs(delta_eV) == 0.0 and abs(dp_eV) == 0.0:
            return cls(_terms=tuple())

        # Convert eV -> J -> (J/hbar) rad/s -> solver
        delta_J = Q(delta_eV, "eV").to("J")
        w_delta = float((delta_J / hbar).to("rad/s").magnitude) * time_unit_s

        dp_J = Q(dp_eV, "eV").to("J")
        w_dp = float((dp_J / hbar).to("rad/s").magnitude) * time_unit_s

        P1 = _proj(qd_i, "X1")
        P2 = _proj(qd_i, "X2")

        # (Delta/2)(P1 - P2)
        H_fss = OpExpr.summation(
            [
                OpExpr.scale(complex(0.5 * w_delta), P1),
                OpExpr.scale(complex(-0.5 * w_delta), P2),
            ]
        )

        # delta_prime (|X1><X2| + |X2><X1|)
        ketbra_x1_x2 = OpExpr.product(
            [_t(qd_i, Transition.G_X1), _t(qd_i, Transition.X2_G)]
        )
        ketbra_x2_x1 = OpExpr.product(
            [_t(qd_i, Transition.G_X2), _t(qd_i, Transition.X1_G)]
        )
        H_mix = OpExpr.scale(
            complex(w_dp),
            OpExpr.summation([ketbra_x1_x2, ketbra_x2_x1]),
        )

        H_total = OpExpr.summation([H_fss, H_mix])

        terms = (
            Term(
                kind=TermKind.H,
                op=H_total,
                coeff=None,
                label="H_exciton_fss_mix",
                meta={
                    "Delta_eV": float(delta_eV),
                    "delta_prime_eV": float(dp_eV),
                    "frame": "rotating_rwa",
                },
            ),
        )
        return cls(_terms=terms)
