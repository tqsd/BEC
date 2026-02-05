from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bec.quantum_dot.dot import QuantumDot


@dataclass(frozen=True)
class DerivedQDBase:
    qd: QuantumDot

    @cached_property
    def t_registry(self):
        # Local import to avoid cycles
        from bec.quantum_dot.transitions import DEFAULT_TRANSITION_REGISTRY

        return getattr(self.qd, "transitions", DEFAULT_TRANSITION_REGISTRY)
