from typing import Optional
import numpy as np
from bec.light.classical import ClassicalTwoPhotonDrive
from bec.quantum_dot.dot import QuantumDot
from bec.simulation.protocols import HamiltonianComposer


class DefaultHamiltonianComposer(HamiltonianComposer):
    def compose(
        self,
        qd: QuantumDot,
        dims: list[int],
        drive: Optional[ClassicalTwoPhotonDrive],
    ) -> list:
        return qd.build_hamiltonians(dims=dims, classical_2g=drive)
