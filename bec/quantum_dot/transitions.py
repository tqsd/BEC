"""
Quantum-dot transition registry (structure + metadata + validation).

This module is the single source of truth for:
- graph topology: endpoints for each directed Transition
- pairing: mapping TransitionPair <-> (forward, backward)
- reverse lookup: Transition -> reverse Transition (O(1))
- metadata/specs: TransitionPair -> TransitionSpec (drive/decay allowed, kind,
  order)

Design goals:
- enums remain "dumb" identifiers (see enums.py)
- registry provides validated, query-friendly access
- QuantumDot can expose a registry instance via qd.transitions (DI-friendly)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Sequence, Tuple

from bec.quantum_dot.enums import (
    QDState,
    Transition,
    TransitionPair,
    TransitionKind,
)


# ---------- Specs (metadata) ----------


@dataclass(frozen=True)
class TransitionSpec:
    """
    Metadata for a transition family (pair).

    kind:
        Broad category (dipole 1-photon, effective 2-photon, ...)

    order:
        Optical order (1 for single-photon dipole, 2 for effective 2-photon, ...)

    drive_allowed:
        Whether coherent drives are allowed to target this transition family.

    decay_allowed:
        Whether radiative decay/collapse terms are allowed for this family.
        (Typically False for effective 2-photon "virtual" transitions.)
    """

    kind: TransitionKind
    order: int
    drive_allowed: bool = True
    decay_allowed: bool = True


# ---------- Registry ----------


@dataclass(frozen=True)
class TransitionRegistry:
    """
    Validated mapping layer for transitions.

    Provides O(1) queries for endpoints/pairing/reverse/specs and ensures the
    structural maps are consistent at construction time.
    """

    _endpoints: Mapping[Transition, Tuple[QDState, QDState]]
    _pair_to_dir: Mapping[TransitionPair, Tuple[Transition, Transition]]
    _specs: Mapping[TransitionPair, TransitionSpec]

    _reverse: Mapping[Transition, Transition]
    _tr_to_pair: Mapping[Transition, TransitionPair]

    @classmethod
    def build(
        cls,
        *,
        endpoints: Mapping[Transition, Tuple[QDState, QDState]],
        pair_to_directed: Mapping[
            TransitionPair, Tuple[Transition, Transition]
        ],
        specs: Mapping[TransitionPair, TransitionSpec],
    ) -> "TransitionRegistry":
        reverse: Dict[Transition, Transition] = {}
        tr_to_pair: Dict[Transition, TransitionPair] = {}

        for pair, (fwd, bwd) in pair_to_directed.items():
            tr_to_pair[fwd] = pair
            tr_to_pair[bwd] = pair
            reverse[fwd] = bwd
            reverse[bwd] = fwd

        reg = cls(
            _endpoints=dict(endpoints),
            _pair_to_dir=dict(pair_to_directed),
            _specs=dict(specs),
            _reverse=reverse,
            _tr_to_pair=tr_to_pair,
        )
        reg.validate()
        return reg

    # ---- validation ----

    def validate(self) -> None:
        # Ensure all directed transitions referenced by pairs exist in endpoints.
        for pair, (fwd, bwd) in self._pair_to_dir.items():
            if fwd not in self._endpoints:
                raise KeyError(f"Missing endpoints for {fwd} (pair {pair})")
            if bwd not in self._endpoints:
                raise KeyError(f"Missing endpoints for {bwd} (pair {pair})")

            a1, b1 = self._endpoints[fwd]
            a2, b2 = self._endpoints[bwd]
            if not (a2 == b1 and b2 == a1):
                raise ValueError(
                    f"Inconsistent pair {pair}: "
                    f"{fwd} endpoints={a1}->{b1} but {bwd} endpoints={a2}->{b2}"
                )

        # Ensure specs exist for every pair.
        missing = [p for p in self._pair_to_dir if p not in self._specs]
        if missing:
            raise KeyError(f"Missing TransitionSpec for pairs: {missing}")

    # ---- core queries ----

    def endpoints(self, tr: Transition) -> Tuple[QDState, QDState]:
        return self._endpoints[tr]

    def reverse(self, tr: Transition) -> Transition:
        return self._reverse[tr]

    def as_pair(self, tr: Transition) -> TransitionPair:
        return self._tr_to_pair[tr]

    def directed(self, tp: TransitionPair) -> Tuple[Transition, Transition]:
        """Return (forward, backward) transitions for a pair."""
        return self._pair_to_dir[tp]

    def pair_endpoints(self, tp: TransitionPair) -> Tuple[QDState, QDState]:
        fwd, _ = self._pair_to_dir[tp]
        return self.endpoints(fwd)

    def spec(self, tr_or_pair: Transition | TransitionPair) -> TransitionSpec:
        if isinstance(tr_or_pair, Transition):
            tr_or_pair = self.as_pair(tr_or_pair)
        return self._specs[tr_or_pair]

    def from_states(self, src: QDState, dst: QDState) -> Optional[Transition]:
        for tr, (a, b) in self._endpoints.items():
            if a == src and b == dst:
                return tr
        return None

    def pairs(self) -> Sequence[TransitionPair]:
        """
        Return registered TransitionPair values in stable enum order.
        """
        present = set(self._pair_to_dir.keys())
        return tuple(tp for tp in TransitionPair if tp in present)


# ---------- Default 4-level QD maps ----------

ENDPOINTS: dict[Transition, Tuple[QDState, QDState]] = {
    Transition.G_X1: (QDState.G, QDState.X1),
    Transition.X1_G: (QDState.X1, QDState.G),
    Transition.G_X2: (QDState.G, QDState.X2),
    Transition.X2_G: (QDState.X2, QDState.G),
    Transition.X1_XX: (QDState.X1, QDState.XX),
    Transition.XX_X1: (QDState.XX, QDState.X1),
    Transition.X2_XX: (QDState.X2, QDState.XX),
    Transition.XX_X2: (QDState.XX, QDState.X2),
    Transition.G_XX: (QDState.G, QDState.XX),
    Transition.XX_G: (QDState.XX, QDState.G),
}

PAIR_TO_DIRECTED: dict[TransitionPair, Tuple[Transition, Transition]] = {
    TransitionPair.G_X1: (Transition.G_X1, Transition.X1_G),
    TransitionPair.G_X2: (Transition.G_X2, Transition.X2_G),
    TransitionPair.X1_XX: (Transition.X1_XX, Transition.XX_X1),
    TransitionPair.X2_XX: (Transition.X2_XX, Transition.XX_X2),
    TransitionPair.G_XX: (Transition.G_XX, Transition.XX_G),
}

SPECS: dict[TransitionPair, TransitionSpec] = {
    TransitionPair.G_X1: TransitionSpec(
        TransitionKind.DIPOLE_1PH, order=1, decay_allowed=True
    ),
    TransitionPair.G_X2: TransitionSpec(
        TransitionKind.DIPOLE_1PH, order=1, decay_allowed=True
    ),
    TransitionPair.X1_XX: TransitionSpec(
        TransitionKind.DIPOLE_1PH, order=1, decay_allowed=True
    ),
    TransitionPair.X2_XX: TransitionSpec(
        TransitionKind.DIPOLE_1PH, order=1, decay_allowed=True
    ),
    # Effective 2-photon excitation "virtual" family: drive yes, decay no
    TransitionPair.G_XX: TransitionSpec(
        TransitionKind.EFFECTIVE_2PH, order=2, decay_allowed=False
    ),
}

DEFAULT_TRANSITION_REGISTRY: TransitionRegistry = TransitionRegistry.build(
    endpoints=ENDPOINTS,
    pair_to_directed=PAIR_TO_DIRECTED,
    specs=SPECS,
)


class RateKey(str, Enum):
    RAD_XX_X1 = "RAD_XX_X1"
    RAD_XX_X2 = "RAD_XX_X2"
    RAD_X1_G = "RAD_X1_G"
    RAD_X2_G = "RAD_X2_G"
    PH_DEPH_X1 = "PH_DEPH_X1"
    PH_DEPH_X2 = "PH_DEPH_X2"
    PH_DEPH_XX = "PH_DEPH_XX"
    PH_RELAX_X1_X2 = "PH_RELAX_X1_X2"
    PH_RELAX_X2_X1 = "PH_RELAX_X2_X1"


RAD_RATE_TO_TRANSITION: dict[RateKey, Transition] = {
    RateKey.RAD_XX_X1: Transition.XX_X1,
    RateKey.RAD_XX_X2: Transition.XX_X2,
    RateKey.RAD_X1_G: Transition.X1_G,
    RateKey.RAD_X2_G: Transition.X2_G,
}
