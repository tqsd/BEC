from __future__ import annotations

from functools import cached_property
from typing import Optional

from bec.quantum_dot.enums import Transition
from bec.quantum_dot.models.phonon_model import PhononModel, PhononOutputs


class PhononsMixin:
    @cached_property
    def phonon_outputs(self) -> PhononOutputs:
        qd = self.qd
        if qd.phonons is None:
            return PhononOutputs()
        return PhononModel(
            phonon_params=qd.phonons,
            transitions=qd.transitions,
        ).compute()

    def polaron_B(self, tr: Transition) -> float:
        return float(self.phonon_outputs.B_polaron_per_transition.get(tr, 1.0))
