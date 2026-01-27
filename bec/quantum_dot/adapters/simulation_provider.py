from __future__ import annotations
from typing import Optional

from bec.simulation.drive_decode import DriveDecodeContext

from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.adapters.polarization_adapter import QDPolarizationAdapter
from bec.quantum_dot.adapters.transition_registry_adapter import (
    QDTransitionRegistryAdapter,
)


class QDDecodeContextProvider:
    def mode_registry(self, qd: QuantumDot):
        return (
            qd.mode_registry
        )  # ensure ModeRegistry implements ModeRegistryView

    def decode_ctx(self, qd: QuantumDot) -> DriveDecodeContext:
        return DriveDecodeContext(
            transitions=QDTransitionRegistryAdapter(qd),
            pol=QDPolarizationAdapter(qd),
            bandwidth=None,
        )

    def derived_for_report(self, qd: QuantumDot) -> Optional[object]:
        return qd.derived

