from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class PulseTrace:
    omega_solver: Optional[np.ndarray] = None
    area_rad: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PlotColumn:
    title: str
    t_solver: np.ndarray
    time_unit_s: float

    # Middle panel: QD populations / observables you want to plot
    qd: Dict[str, np.ndarray]  # e.g. {"P_G":..., "P_X1":..., ...}

    # Bottom panel: photonic outputs grouped by label; each label can have H/V
    outputs_H: Dict[str, np.ndarray]  # e.g. {"G_X":..., "X_XX":...}
    outputs_V: Dict[str, np.ndarray]

    # Optional top panel
    pulse: Optional[PulseTrace] = None

    # Optional extras if you want later
    meta: Optional[dict] = None
