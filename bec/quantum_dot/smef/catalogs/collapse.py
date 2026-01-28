from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.terms import Term, TermKind
from smef.core.model.protocols import TermCatalogProto

from bec.quantum_dot.enums import RateKey, Transition, TransitionPair
from bec.quantum_dot.transitions import RAD_RATE_TO_TRANSITION

from bec.quantum_dot.smef.catalogs.base import FrozenCatalog


def _symbol_for_directed_transition(tr: Transition) -> str:
    # Transition is "SRC_DST" for SRC -> DST.
    # Collapse operator for spontaneous emission uses |DST><SRC|.
    src, dst = tr.value.split("_", 1)
    return f"t_{dst}_{src}"


@dataclass(frozen=True)
class QDCollapseCatalog(FrozenCatalog):
    @classmethod
    def from_qd(cls, qd, *, units) -> TermCatalogProto:
        """
        Build radiative collapse operators.

        Requires qd.derived to provide:
          - rates_rad_ratekey(): mapping RateKey -> QuantityLike (1/s)
        OR you can replace that with however your RatesMixin exposes them.
        """
        derived = qd.derived
        modes = qd.smef_modes if hasattr(qd, "smef_modes") else None

        qd_index = 0  # qd subsystem index in QDModes
        terms: list[Term] = []

        # You decide where rates live. This is the only “contract” point.
        rates = getattr(derived, "rates", None)
        if rates is None:
            # common alternative naming
            rates = getattr(derived, "rates_rad", None)

        if rates is None:
            # As a fallback, allow qd to provide them directly
            rates = getattr(qd, "rates", None)

        if rates is None:
            raise AttributeError(
                "No rates found. Expected derived.rates (or derived.rates_rad or qd.rates)."
            )

        def rate_value(rate_key: RateKey) -> float:
            r = (
                rates[rate_key]
                if isinstance(rates, dict)
                else rates.get(rate_key)
            )
            # converts 1/s -> solver 1/(solver unit)
            return float(units.rate_to_solver(r))

        for rk, tr in RAD_RATE_TO_TRANSITION.items():
            pair = qd.transitions.as_pair(tr)
            spec = qd.transitions.spec(pair)
            if not spec.decay_allowed:
                continue

            gamma_solver = rate_value(rk)
            if gamma_solver < 0.0:
                raise ValueError(
                    f"Negative decay rate for {
                                 rk}: {gamma_solver}"
                )

            pref = complex(np.sqrt(gamma_solver))
            sym = _symbol_for_directed_transition(tr)

            op = OpExpr.scale(
                pref,
                OpExpr.atom(
                    EmbeddedKron(
                        indices=(qd_index,),
                        locals=(LocalSymbolOp(sym),),
                    )
                ),
            )

            terms.append(
                Term(
                    kind=TermKind.C,
                    op=op,
                    coeff=None,
                    label=f"L_{rk.value}",
                    meta={"rate_key": rk.value, "transition": tr.value},
                )
            )

        return cls(_terms=tuple(terms))
