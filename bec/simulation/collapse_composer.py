from qutip import Qobj
from bec.quantum_dot.dot import QuantumDot
from bec.simulation.protocols import CollapseComposer


class DefaultCollapseComposer(CollapseComposer):
    def compose(self, qd: QuantumDot, dims: list[int]) -> list[Qobj]:
        return qd.qutip_collapse_operators(dims)
