from __future__ import annotations

from typing import Any, Union, Optional, Dict
from enum import Enum
from dataclasses import dataclass, field

import numpy as np

from .coeffs import CoeffExpr, ConstCoeff


class HamiltonianTermKind(str, Enum):
    STATIC = "static"
    DRIVE = "drive"
    DETUNING = "detuning"
    INTERACTION = "interaction"


class CollapseTermKind(str, Enum):
    RADIATIVE = "radiative"
    PHONON = "phonon"
    PHENOMENOLOGICAL = "phenomenological"


@dataclass
class HamiltonianTerm:
    kind: HamiltonianTermKind
    op: np.ndarray
    coeff: Optional[CoeffExpr] = None
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollapseTerm:
    kind: CollapseTermKind
    op: np.ndarray
    coeff: Optional[CoeffExpr] = None
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observables:
    qd: Dict[str, np.ndarray]
    modes: Dict[str, np.ndarray]
    extra: Optional[Dict[str, np.ndarray]] = None
