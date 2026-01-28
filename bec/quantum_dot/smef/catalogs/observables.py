from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr, OpExprKind
from smef.core.ir.terms import Term, TermKind

from bec.quantum_dot.ops.symbols import QDSymbol, ModeSymbol, sym
from bec.quantum_dot.smef.modes import QDModeKey, QDModes


@dataclass(frozen=True)
class QDObservablesCatalog:
    modes: QDModes

    @property
    def all_terms(self) -> Sequence[Term]:
        qd_i = self.modes.index_of(QDModeKey.qd())

        # QD populations
        pops = [
            Term(
                kind=TermKind.E,
                label="pop_G",
                op=OpExpr.atom(
                    EmbeddedKron(
                        indices=(qd_i,),
                        locals=(LocalSymbolOp(sym(QDSymbol.PROJ_G)),),
                    )
                ),
            ),
            Term(
                kind=TermKind.E,
                label="pop_X1",
                op=OpExpr.atom(
                    EmbeddedKron(
                        indices=(qd_i,),
                        locals=(LocalSymbolOp(sym(QDSymbol.PROJ_X1)),),
                    )
                ),
            ),
            Term(
                kind=TermKind.E,
                label="pop_X2",
                op=OpExpr.atom(
                    EmbeddedKron(
                        indices=(qd_i,),
                        locals=(LocalSymbolOp(sym(QDSymbol.PROJ_X2)),),
                    )
                ),
            ),
            Term(
                kind="E",
                label="pop_XX",
                op=OpExpr.atom(
                    EmbeddedKron(
                        indices=(qd_i,),
                        locals=(LocalSymbolOp(sym(QDSymbol.PROJ_XX)),),
                    )
                ),
            ),
        ]

        # Photon numbers per mode (optional but very handy)
        n_terms = []
        for key, label in [
            (QDModeKey.gx("H"), "n_GX_H"),
            (QDModeKey.gx("V"), "n_GX_V"),
            (QDModeKey.xx("H"), "n_XX_H"),
            (QDModeKey.xx("V"), "n_XX_V"),
        ]:
            mi = self.modes.index_of(key)
            n_terms.append(
                Term(
                    kind="E",
                    label=label,
                    op=OpExpr.atom(
                        EmbeddedKron(
                            indices=(mi,),
                            locals=(LocalSymbolOp(sym(ModeSymbol.N)),),
                        )
                    ),
                )
            )

        return tuple(pops + n_terms)
