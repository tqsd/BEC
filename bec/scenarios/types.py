from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class SchemeKind(str, Enum):
    TPE = "tpe"
    ARP = "arp"
    BICHROMATIC = "bichromatic"


class SweepAxis(str, Enum):
    AMP_SCALE = "amp_scale"
    DETUNING_OFFSET_RAD_S = "detuning_offset_rad_s"


DriveFactory = Callable[..., Tuple[List[Any], List[Any]]]
# Returns: (drive_specs, payloads_for_plotting)


@dataclass(frozen=True)
class ScenarioRunConfig:
    t_end_ns: float = 2.0
    n_points: int = 2001
    t0_ns: float = 1.0
    sigma_ns: float = 0.05
    audit: bool = False
    compensate_polaron: bool = True


@dataclass(frozen=True)
class ScenarioResult:
    scheme: SchemeKind
    x_params: Mapping[str, float]
    # raw objects (keep for plotting/debug)
    res: Any
    metrics: Any
    tlist: np.ndarray
    units: Any
    drives: Sequence[Any] = field(default_factory=tuple)

    # extracted scalars for sweeps
    xx_final: float = 0.0
    # you can extend this later:
    # bell_fidelity: float = 0.0
    # log_negativity: float = 0.0


@dataclass(frozen=True)
class SweepSpec:
    axis: SweepAxis
    values: np.ndarray
    threshold_xx: float = 0.9


@dataclass(frozen=True)
class SweepGridSpec:
    x_axis: SweepSpec
    y_axis: SweepSpec
    threshold_xx: float = 0.9


@dataclass(frozen=True)
class Sweep1DResult:
    scheme: SchemeKind
    axis: SweepAxis
    values: np.ndarray
    xx_final: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sweep2DResult:
    scheme: SchemeKind
    x_axis: SweepAxis
    y_axis: SweepAxis
    x_values: np.ndarray
    y_values: np.ndarray
    xx_final: np.ndarray  # shape (len(y), len(x))
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobustnessSummary:
    scheme: SchemeKind
    metric: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
