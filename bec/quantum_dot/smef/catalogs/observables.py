from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.terms import Term, TermKind

from bec.quantum_dot.ops.symbols import QDSymbol, ModeSymbol, sym
from bec.quantum_dot.smef.modes import QDModeKey, QDModes


@dataclass(frozen=True)
class QDObservablesCatalog:
    modes: QDModes
    include_coherences: bool = False
    include_two_photon_coherence: bool = True

    @property
    def all_terms(self) -> Sequence[Term]:
        qd_i = self.modes.index_of(QDModeKey.qd())

        def _qd_local(symbol: str) -> OpExpr:
            return OpExpr.atom(
                EmbeddedKron(
                    indices=(qd_i,),
                    locals=(LocalSymbolOp(symbol),),
                )
            )

        # QD populations
        pops = [
            Term(
                kind=TermKind.E,
                label="pop_G",
                op=_qd_local(sym(QDSymbol.PROJ_G)),
            ),
            Term(
                kind=TermKind.E,
                label="pop_X1",
                op=_qd_local(sym(QDSymbol.PROJ_X1)),
            ),
            Term(
                kind=TermKind.E,
                label="pop_X2",
                op=_qd_local(sym(QDSymbol.PROJ_X2)),
            ),
            Term(
                kind=TermKind.E,
                label="pop_XX",
                op=_qd_local(sym(QDSymbol.PROJ_XX)),
            ),
        ]

        # Photon numbers per mode
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
                    kind=TermKind.E,
                    label=label,
                    op=OpExpr.atom(
                        EmbeddedKron(
                            indices=(mi,),
                            locals=(LocalSymbolOp(sym(ModeSymbol.N)),),
                        )
                    ),
                )
            )

        coh_terms = []
        if self.include_coherences:
            # Complex coherences (expect may be complex). Plotter can use abs/re/im.
            coh_terms.extend(
                [
                    Term(
                        kind=TermKind.E,
                        label="coh_G_X1",
                        op=_qd_local(sym(QDSymbol.T_G_X1)),
                    ),
                    Term(
                        kind=TermKind.E,
                        label="coh_G_X2",
                        op=_qd_local(sym(QDSymbol.T_G_X2)),
                    ),
                ]
            )
            if self.include_two_photon_coherence:
                # Choose ONE depending on your convention:
                # - T_G_XX: direct |G><XX| operator
                # - SX_G_XX: effective 2ph coupling operator used in Hamiltonian
                coh_terms.append(
                    Term(
                        kind=TermKind.E,
                        label="coh_G_XX",
                        op=_qd_local(sym(QDSymbol.T_G_XX)),
                    )
                )

        return tuple(pops + n_terms + coh_terms)
