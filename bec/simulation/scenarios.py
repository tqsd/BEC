from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable
from bec.light.classical import ClassicalTwoPhotonDrive
from bec.light.light_mode import LightMode
from bec.quantum_dot import QuantumDot
from .protocols import Scenario


@dataclass
class ClassicalDriveScenario(Scenario):
    drive: ClassicalTwoPhotonDrive
    name: str = "classical-2g"

    def prepare(self, qd: QuantumDot) -> None:
        # No external modes registered; only classical 2γ
        return

    def classical_drive(self) -> Optional[ClassicalTwoPhotonDrive]:
        return self.drive

    def label(self) -> str:
        return self.name


@dataclass
class QuantumInputsScenario(Scenario):
    """Not used by the current model"""

    inputs: Iterable[LightMode]
    name: str = "quantum-inputs"

    def prepare(self, qd: QuantumDot) -> None:
        for lm in self.inputs:
            qd.register_flying_mode(light_mode=lm)

    def classical_drive(self) -> Optional[ClassicalTwoPhotonDrive]:
        return None

    def label(self) -> str:
        return self.name


@dataclass
class HybridScenario(Scenario):
    """Not used by the current model"""

    inputs: Iterable[LightMode]
    drive: ClassicalTwoPhotonDrive
    name: str = "hybrid"

    def prepare(self, qd: QuantumDot) -> None:
        for lm in self.inputs:
            qd.register_flying_mode(light_mode=lm)

    def classical_drive(self) -> Optional[ClassicalTwoPhotonDrive]:
        return self.drive

    def label(self) -> str:
        return self.name
