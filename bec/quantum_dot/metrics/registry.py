from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from qutip import Qobj


@dataclass(frozen=True)
class PhotonicRegistry:
    """Immutable snapshot of the photonic layout used by diagnostics/metrics."""

    dims_phot: List[int]
    early_factors: List[int]
    late_factors: List[int]
    offset: int
    labels_by_mode_index: List[str]
    number_op_by_factor: Dict[int, Qobj]
    proj0_by_factor: Dict[int, Qobj]
    proj1_by_factor: Dict[int, Qobj]
    I_phot: Qobj

    @property
    def Dp(self) -> int:
        return int(np.prod(self.dims_phot))
