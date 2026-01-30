from __future__ import annotations

from typing import Any

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr

from bec.quantum_dot.enums import QDState, Transition
from bec.quantum_dot.ops.symbols import QDSymbol, sym


def qd_local(qd_index: int, symbol: Any) -> OpExpr:
    """
    Embed a QD-local operator symbol on the QD subsystem at index qd_index.
    """
    return OpExpr.atom(
        EmbeddedKron(
            indices=(int(qd_index),), locals=(LocalSymbolOp(sym(symbol)),)
        )
    )


def proj_symbol(state: QDState) -> str:
    if state is QDState.G:
        return sym(QDSymbol.PROJ_G)
    if state is QDState.X1:
        return sym(QDSymbol.PROJ_X1)
    if state is QDState.X2:
        return sym(QDSymbol.PROJ_X2)
    if state is QDState.XX:
        return sym(QDSymbol.PROJ_XX)
    raise KeyError(state)


def transition_symbol(tr: Transition) -> str:
    """
    Your operator names include "t_" + Transition.value
    """
    return "t_" + str(tr.value)
