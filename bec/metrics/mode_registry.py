from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from bec.metrics.linops import _as_list_int


@dataclass(frozen=True)
class MetricGroups:
    """
    Semantic grouping for metrics.

    Convention (can change if you want):
      early = XX band (XX -> X photon)
      late  = GX band (X -> G photon)
    """

    qd: Sequence[Hashable]
    gx: Sequence[Hashable]
    xx: Sequence[Hashable]

    @property
    def early(self) -> Sequence[Hashable]:
        return self.xx

    @property
    def late(self) -> Sequence[Hashable]:
        return self.gx

    @property
    def gx_pol(self) -> Sequence[Hashable]:
        return self.gx

    @property
    def xx_pol(self) -> Sequence[Hashable]:
        return self.xx


def default_groups(modes: Any) -> MetricGroups:
    """
    Build default metric groups from a QDModes-like registry.

    modes must support index_of(key) and accept the string shortcuts you defined.
    """
    return MetricGroups(
        qd=("qd",),
        gx=("GX_H", "GX_V"),
        xx=("XX_H", "XX_V"),
    )


def indices_of(modes: Any, keys: Iterable[Hashable]) -> list[int]:
    return _as_list_int([modes.index_of(k) for k in keys])
