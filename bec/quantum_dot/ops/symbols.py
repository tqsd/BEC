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

    T_XX_G = "t_XX_G"
    T_G_XX = "t_G_XX"
    SX_G_XX = "sx_G_XX"


class ModeSymbol(str, Enum):
    A = "a"
    ADAG = "adag"
    N = "n"


SymbolLike = Union[str, QDSymbol, ModeSymbol]


def sym(x: SymbolLike) -> str:
    # Normalize to the canonical string used in SymbolOp/LocalSymbolOp
    if isinstance(x, Enum):
        return str(x.value)
    return str(x)
