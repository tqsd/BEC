from __future__ import annotations

from functools import cached_property

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
            exciton_splitting=qd.mixing,
        ).compute()

    def polaron_B(self, tr: Transition) -> float:
        return float(self.phonon_outputs.B_polaron_per_transition.get(tr, 1.0))

    @property
    def gamma_phi_eid_scale(self) -> float:
        qd = self.qd
        P = getattr(qd, "phonons", None)
        if P is None:
            return 0.0
        ph = getattr(P, "phenomenological", None)
        if ph is None:
            return 0.0
        return float(getattr(ph, "gamma_phi_eid_scale", 0.0) or 0.0)

    @property
    def polaron_B_X(self) -> float:
        return 0.5 * (
            self.polaron_B(Transition.G_X1) + self.polaron_B(Transition.G_X2)
        )

    @property
    def polaron_B_XX(self) -> float:
        return 0.5 * (
            self.polaron_B(Transition.X1_XX) + self.polaron_B(Transition.X2_XX)
        )
