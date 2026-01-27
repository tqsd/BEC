from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.me.types import Observables


@dataclass(frozen=True)
class ObservablesCompositionPolicy:
    """
    include_qd:
        Include QD projectors (P_G, P_X1, ...).

    include_modes:
        Include mode observables (N, N+, N-, Pvac, P10, ...).

    include_qd_in_mode_ops:
        Kept for future compatibility (you removed reduction).
        Currently has no effect unless you reintroduce reduction logic.
    """

    include_qd: bool = True
    include_modes: bool = True
    include_qd_in_mode_ops: bool = True  # placeholder / future


class DefaultObservablesComposer:
    def __init__(self, policy: Optional[ObservablesCompositionPolicy] = None):
        self.policy = policy or ObservablesCompositionPolicy()

    def compose(self, qd: QuantumDot, dims: List[int]) -> Observables:
        obs = qd.o_builder.build(
            dims, include_qd=self.policy.include_qd_in_mode_ops
        )

        if not self.policy.include_qd:
            obs = Observables(
                qd={},
                modes=obs.modes if self.policy.include_modes else {},
                extra=obs.extra,
            )

        elif not self.policy.include_modes:
            obs = Observables(qd=obs.qd, modes={}, extra=obs.extra)

        return obs
