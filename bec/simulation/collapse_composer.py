from qutip import Qobj
from bec.quantum_dot.dot import QuantumDot
from bec.simulation.protocols import CollapseComposer


class DefaultCollapseComposer(CollapseComposer):
    def compose(
        self, qd: QuantumDot, dims: list[int], time_unit_s: float = 1.0
    ) -> list[Qobj]:
        return qd.qutip_collapse_operators(dims, time_unit_s)
