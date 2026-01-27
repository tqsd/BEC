from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from bec.quantum_dot.dot import QuantumDot

from bec.quantum_dot.enums import Transition


class QDTransitionRegistryAdapter:
    def __init__(self, qd: "QuantumDot"):
        self._qd = qd

    def transitions(self) -> Tuple[Transition, ...]:
        # the ones you want to consider for decoding
        return (
            Transition.G_X1,
            Transition.G_X2,
            Transition.X1_XX,
            Transition.X2_XX,
            Transition.G_XX,
        )

    def kind(self, tr: Transition) -> str:
        # hardcode or derive from qd.t_registry/spec if you prefer
        return "2ph" if tr is Transition.G_XX else "1ph"

    def omega_ref_rad_s(self, tr: Transition) -> float:
        return float(self._qd.derived.omega_ref_rad_s(tr))
