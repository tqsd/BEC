from typing import Dict
from qutip import Qobj
from bec.quantum_dot.dot import QuantumDot
from bec.simulation.protocols import ObservableComposer


class DefaultObservableComposer(ObservableComposer):
    def compose_qd(
        self, qd: QuantumDot, dims: list[int], time_unit_s: float
    ) -> Dict[str, Qobj]:
        return qd.qutip_projectors(dims)

    def compose_lm(
        self, qd: QuantumDot, dims: list[int], time_unit_s: float
    ) -> Dict[str, Qobj]:
        return qd.qutip_light_mode_projectors(dims)
