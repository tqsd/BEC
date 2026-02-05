from __future__ import annotations

from enum import Enum
from typing import Union


class QDSymbol(str, Enum):
    PROJ_G = "proj_G"
    PROJ_X1 = "proj_X1"
    PROJ_X2 = "proj_X2"
    PROJ_XX = "proj_XX"

    T_X1_XX = "t_X1_XX"
    T_X2_XX = "t_X2_XX"
    T_G_X1 = "t_G_X1"
    T_G_X2 = "t_G_X2"
    T_X1_X2 = "t_X1_X2"
    T_X2_X1 = "t_X2_X1"

    T_XX_G = "t_XX_G"
    T_G_XX = "t_G_XX"
    SX_G_XX = "sx_G_XX"


class ModeSymbol(str, Enum):
    A = "a"
    ADAG = "adag"
    N = "n"
    I = "I"


SymbolLike = Union[str, QDSymbol, ModeSymbol]


def sym(x: SymbolLike) -> str:
    # Normalize to the canonical string used in SymbolOp/LocalSymbolOp
    if isinstance(x, Enum):
        return str(x.value)
    return str(x)


def qd_symbol_latex_map() -> dict[str, str]:
    # Canonical symbol string -> LaTeX
    return {
        # Projectors
        sym(QDSymbol.PROJ_G): r"\vert G \rangle\langle G \vert",
        sym(QDSymbol.PROJ_X1): r"\vert X_1 \rangle\langle X_1 \vert",
        sym(QDSymbol.PROJ_X2): r"\vert X_2 \rangle\langle X_2 \vert",
        sym(QDSymbol.PROJ_XX): r"\vert XX \rangle\langle XX \vert",
        # Transitions: your t_SRC_DST corresponds to |DST><SRC|
        sym(QDSymbol.T_G_X1): r"\hat{\sigma}_{X_1,G}",
        sym(QDSymbol.T_G_X2): r"\hat{\sigma}_{X_2,G}",
        sym(QDSymbol.T_X1_XX): r"\hat{\sigma}_{XX,X_1}",
        sym(QDSymbol.T_X2_XX): r"\hat{\sigma}_{XX,X_2}",
        sym(QDSymbol.T_XX_G): r"\hat{\sigma}_{G,XX}",
        sym(QDSymbol.T_G_XX): r"\hat{\sigma}_{XX,G}",
        sym(QDSymbol.T_X1_X2): r"\hat{\sigma}_{X_2,X_1}",
        sym(QDSymbol.T_X2_X1): r"\hat{\sigma}_{X_1,X_2}",
        # Convenience operator
        sym(QDSymbol.SX_G_XX): r"\hat{\sigma}_{XX,G} + \hat{\sigma}_{G,XX}",
        # Mode operators
        sym(ModeSymbol.A): r"\hat{a}",
        sym(ModeSymbol.ADAG): r"\hat{a}^{\dagger}",
        sym(ModeSymbol.N): r"\hat{n}",
        sym(ModeSymbol.I): r"\mathbb{I}",
    }
