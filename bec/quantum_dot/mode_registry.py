from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Tuple

from bec.light.core.channels import LightChannel
from bec.units import as_quantity, energy_to_wavelength
from bec.quantum_dot.enums import QDState, Transition
from bec.quantum_dot.transitions import (
    DEFAULT_REGISTRY as DEFAULT_TRANSITION_REGISTRY,
    TransitionRegistry,
)
from bec.quantum_dot.parameters.energy_structure import EnergyStructure

Pol = Literal["H", "V"]


class SpectralResolution(str, Enum):
    TWO_COLORS = "two_colors"
    FULLY_RESOLVED = "fully_resolved"


@dataclass(frozen=True)
class ChannelKey:
    kind: str
    pol: Pol


def channel_kind_for_transition(
    reg: TransitionRegistry, tr: Transition, *, resolution: SpectralResolution
) -> str:
    tp = reg.as_pair(tr)
    if resolution == SpectralResolution.FULLY_RESOLVED:
        return tp.value

    a, b = reg.pair_endpoints(tp)
    if {a, b} in ({QDState.G, QDState.X1}, {QDState.G, QDState.X2}):
        return "X->G"
    if {a, b} in ({QDState.X1, QDState.XX}, {QDState.X2, QDState.XX}):
        return "XX->X"
    if {a, b} == {QDState.G, QDState.XX}:
        return "G->XX(virtual)"
    return "other"


def _channel_kinds_for_build(
    reg: TransitionRegistry, resolution: SpectralResolution
) -> List[str]:
    if resolution == SpectralResolution.TWO_COLORS:
        return ["XX->X", "X->G"]

    # FULLY_RESOLVED: one per *radiative* 1-ph family
    kinds: List[str] = []
    for (
        tp,
        spec,
    ) in reg._specs.items():  # add a reg.pairs() accessor later if you want
        if spec.order == 1 and spec.decay_allowed:
            kinds.append(tp.value)
    return kinds


def _energy_for_kind(kind: str, es: EnergyStructure) -> "QuantityLike | None":
    # return a representative photon energy for that channel
    # assumes EnergyStructure has X1, X2, XX and optionally exciton_center
    X1 = as_quantity(es.X1, "eV")
    X2 = as_quantity(es.X2, "eV")
    XX = as_quantity(es.XX, "eV")
    xc = getattr(es, "exciton_center", None)
    if xc is None:
        xc = (X1 + X2) / 2.0
    else:
        xc = as_quantity(xc, "eV")

    if kind == "X->G":
        return xc
    if kind == "XX->X":
        return (XX - xc).to("eV")

    # FULLY_RESOLVED kinds use tp.value strings like "G_X1"
    if kind == "G_X1":
        return X1
    if kind == "G_X2":
        return X2
    if kind == "X1_XX":
        return (XX - X1).to("eV")
    if kind == "X2_XX":
        return (XX - X2).to("eV")

    return None


class ModeRegistry:
    def __init__(
        self,
        *,
        channels: List[LightChannel],
        resolution: SpectralResolution,
        reg: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY,
    ):
        self._channels = channels
        self._reg = reg
        self.resolution = resolution
        self._index: Dict[ChannelKey, int] = {
            ch.key: i for i, ch in enumerate(channels)
        }

    @classmethod
    def from_qd(
        cls,
        *,
        energy_structure: EnergyStructure,
        transitions: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY,
        resolution: SpectralResolution = SpectralResolution.TWO_COLORS,
    ) -> "ModeRegistry":
        channels: List[LightChannel] = []
        for kind in _channel_kinds_for_build(transitions, resolution):
            E = _energy_for_kind(kind, energy_structure)
            for pol in ("H", "V"):
                key = ChannelKey(kind=kind, pol=pol)
                channels.append(
                    LightChannel(
                        key=key,
                        energy=E,
                        wavelength=energy_to_wavelength(E, out_unit="nm"),
                        label=f"{kind}:{pol}",
                        label_tex=None,
                    )
                )
        return cls(channels=channels, reg=transitions, resolution=resolution)

    @property
    def channels(self) -> List[LightChannel]:
        return self._channels

    def num_modes(self) -> int:
        return len(self._channels)

    def by_transition(
        self, tr: Transition, pol: Pol
    ) -> Tuple[int, LightChannel]:
        kind = channel_kind_for_transition(
            self._reg, tr, resolution=self.resolution
        )
        key = ChannelKey(kind=kind, pol=pol)
        i = self._index[key]
        return i, self._channels[i]

    def index_of(self, key: ChannelKey) -> int:
        return self._index[key]
