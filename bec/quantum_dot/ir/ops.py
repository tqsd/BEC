from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any

import numpy as np

from bec.quantum_dot.mode_registry import ChannelKey


class FockOpKind(str, Enum):
    A = "a"
    ADAG = "adag"
    N = "n"
    VAC = "vac"


@dataclass(frozen=True)
class QDOpRef:
    """
    Reference to a QD operator.
    - key: resolved via context (e.g. "s_X1_G")
    - or mat: directly embedded numeric operator
    """

    key: Optional[str] = None
    mat: Optional[np.ndarray] = None

    def __post_init__(self):
        if (self.key is None) == (self.mat is None):
            raise ValueError("QDOpRef requires exactly one of (key, mat).")


@dataclass(frozen=True)
class FockOpRef:
    """
    Photonic operator acting on a particular logical channel (mode index + pol).
    """

    key: ChannelKey
    kind: FockOpKind  # a, adag, n, vac, I
    label: Optional[str] = None


@dataclass(frozen=True)
class EmbeddedKron:
    """
    A 'kron(QD, mode0, mode1, ...)' embedding request, but typed.

    Exactly one non-identity fock op is typical, but not required.
    """

    qd: QDOpRef
    fock: Optional[FockOpRef] = None

    meta: Dict[str, Any] = field(default_factory=dict)

    # placeholder for future pretty printing
    pretty: Optional[str] = None


class OpExprKind(Enum):
    PRIMITIVE = "primitive"
    SUM = "sum"
    PROD = "prod"
    SCALE = "smult"


@dataclass(frozen=True)
class OpExpr:
    kind: OpExprKind
    args: tuple["OpExpr", ...] = ()
    primitive: Optional[EmbeddedKron] = None
    scalar: Any = None
    pretty: Optional[str] = None
