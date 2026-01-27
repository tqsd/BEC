from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, TypeAlias

import numpy as np

TransitionKey: TypeAlias = Any


class TransitionRegistryView(Protocol):
    def transitions(self) -> Tuple[TransitionKey, ...]: ...
    def kind(self, tr: TransitionKey) -> str: ...  # "1ph"/"2ph"
    def omega_ref_rad_s(self, tr: TransitionKey) -> float: ...  # physical rad/s


class PolarizationCouplingView(Protocol):
    def coupling_weight(
        self, tr: TransitionKey, E_hv: np.ndarray
    ) -> complex: ...


class BandwidthEstimator(Protocol):
    def sigma_omega_rad_s(
        self, *, drive: Any, tlist_solver: np.ndarray, time_unit_s: float
    ) -> float: ...


class PhononRenormView(Protocol):
    def polaron_B(self) -> float: ...


@dataclass(frozen=True)
class DriveDecodeContext:
    transitions: TransitionRegistryView
    pol: Optional[PolarizationCouplingView] = None
    bandwidth: Optional[BandwidthEstimator] = None
    phonons: Optional[PhononRenormView] = None
