from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DriveSeries:
    label: str
    t_s: np.ndarray
    E_env_V_m: Optional[np.ndarray] = None
    omega_L_rad_s: Optional[np.ndarray] = None
    delta_omega_rad_s: Optional[np.ndarray] = None
    Omega_rad_s: Optional[np.ndarray] = None  # drive-implied mu*E/hbar


@dataclass(frozen=True)
class QDTraces:
    t_solver: np.ndarray
    t_s: np.ndarray
    time_unit_s: float

    pops: Mapping[str, np.ndarray] = field(default_factory=dict)
    outputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    coherences: Mapping[str, np.ndarray] = field(default_factory=dict)

    drives: Sequence[DriveSeries] = field(default_factory=tuple)

    extra: Mapping[str, np.ndarray] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
