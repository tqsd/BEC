from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.terms import Term, TermKind
from smef.core.model.protocols import TermCatalogProto

from smef.core.units import Q, hbar, magnitude

from bec.quantum_dot.enums import Transition
from bec.quantum_dot.smef.catalogs.base import FrozenCatalog
from bec.quantum_dot.smef.modes import QDModes, QDModeKey


def _qd_local(qd_index: int, symbol: str) -> OpExpr:
    return OpExpr.atom(
        EmbeddedKron(indices=(qd_index,), locals=(LocalSymbolOp(symbol),))
    )


def _proj_X1(qd_index: int) -> OpExpr:
    return _qd_local(qd_index, "proj_X1")


def _proj_X2(qd_index: int) -> OpExpr:
    return _qd_local(qd_index, "proj_X2")


def _t(qd_index: int, tr: Transition) -> OpExpr:
    # qd4_named_ops registers "t_" + Transition.value as an alias
    return _qd_local(qd_index, "t_" + str(tr.value))


def _delta_prime_eV(qd) -> float:
    mp = getattr(qd, "exciton_mixing_params", None)
    if mp is None:
        return 0.0
    dp = getattr(mp, "delta_prime", 0.0)
    try:
        return float(magnitude(dp, "eV"))
    except Exception:
        return float(dp)


@dataclass(frozen=True)
class QDHamiltonianCatalog(FrozenCatalog):
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

        # Dot subsystem index
        qd_i = 0 if modes is None else int(modes.index_of(QDModeKey.qd()))

        # FSS: Delta = E(X1) - E(X2) in eV
        E_X1_eV = float(magnitude(qd.energy.X1, "eV"))
        E_X2_eV = float(magnitude(qd.energy.X2, "eV"))
        Delta_eV = E_X1_eV - E_X2_eV

        # Convert energies -> angular frequency (rad/s) -> solver units (multiply by time_unit_s)
        time_unit_s = float(units.time_unit_s)

        Delta_J = Q(Delta_eV, "eV").to("J")
        w_Delta_rad_s = float((Delta_J / hbar).to("rad/s").magnitude)
        w_Delta_solver = w_Delta_rad_s * time_unit_s

        dp_eV = _delta_prime_eV(qd)
        dp_J = Q(dp_eV, "eV").to("J")
        w_dp_rad_s = float((dp_J / hbar).to("rad/s").magnitude)
        w_dp_solver = w_dp_rad_s * time_unit_s

        # Operators in the QD subsystem
        P1 = _proj_X1(qd_i)
        P2 = _proj_X2(qd_i)

        # Diagonal FSS part: (Delta/2)(P_X1 - P_X2)
        H_fss = OpExpr.summation(
            [
                OpExpr.scale(0.5 * complex(w_Delta_solver), P1),
                OpExpr.scale(-0.5 * complex(w_Delta_solver), P2),
            ]
        )

        # Mixing part: dp (|X1><X2| + |X2><X1|)
        #
        # Using only existing transition symbols with your convention:
        # t_SRC_DST = |DST><SRC|
        #
        # |X1><X2| = (|X1><G|)(|G><X2|) = t_G_X1 * t_X2_G
        # |X2><X1| = (|X2><G|)(|G><X1|) = t_G_X2 * t_X1_G
        ketbra_X1_X2 = OpExpr.product(
            [_t(qd_i, Transition.G_X1), _t(qd_i, Transition.X2_G)]
        )
        ketbra_X2_X1 = OpExpr.product(
            [_t(qd_i, Transition.G_X2), _t(qd_i, Transition.X1_G)]
        )

        H_mix = OpExpr.scale(
            complex(w_dp_solver),
            OpExpr.summation([ketbra_X1_X2, ketbra_X2_X1]),
        )

        # Total exciton Hamiltonian in rotating frame (slow terms only)
        H_total = OpExpr.summation([H_fss, H_mix])

        terms = (
            Term(
                kind=TermKind.H,
                op=H_total,
                coeff=None,
                label="H_exciton_fss_mix",
                meta={
                    "Delta_eV": float(Delta_eV),
                    "delta_prime_eV": float(dp_eV),
                    "frame": "rotating_rwa",
                },
            ),
        )

        return cls(_terms=terms)
