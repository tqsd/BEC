from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Literal, Optional


from bec.quantum_dot.ir.ops import OpExpr
from bec.quantum_dot.mode_registry import ChannelKey


Pol = Literal["H", "V"]  # or reuse your light.types.Pol


class CoeffRefKind(str, Enum):
    CONST = "const"
    RATE = "rate"
    DRIVE = "drive"
    PARAM = "param"
    EXPR = "expr"


@dataclass(frozen=True)
class CoeffRef:
    kind: CoeffRefKind
    key: Optional[str] = None
    value: Any | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind == CoeffRefKind.CONST:
            if self.value is None:
                raise ValueError("CONST coeff must define value.")
        else:
            if not self.key:
                raise ValueError(f"{self.kind} coeff must define key.")


class TermKind(Enum):
    H = "Hamiltonian"
    C = "Collapse"
    E = "Observable"


@dataclass(frozen=True)
class Term:
    kind: TermKind
    label: str
    op: OpExpr
    coeff: Optional[CoeffRef] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    pretty: Optional[str] = None
