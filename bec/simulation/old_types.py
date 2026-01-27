from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from bec.light.classical import ClassicalCoherentDrive
from bec.params.transitions import Transition
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.me.coeffs import CoeffExpr
from bec.quantum_dot.me.types import (
    CollapseTerm,
    HamiltonianTerm,
    Observables,
)


@dataclass(frozen=True)
class MEProblem:
    """
    Backend-agnostic compiled master-equation problem

    Stores the inremediate representation terms and can
    export to QuTiP format via helper properties

    Notes:
    ------
    - Hamiltonian.coeff may be None (static) or CoeffExpr callable
    - CollapseTerm.coeff must be set; ConstCoeff will be folded into op
      by c_terms_to_qutip, otherwise returned as [op, coeff]
    """

    qd: QuantumDot
    tlist: np.ndarray
    time_unit_s: float

    # Hilbert space (QuTiP dims list) and initial state
    dims: List[int]
    rho0: np.ndarray

    h_terms: Tuple[HamiltonianTerm, ...] = field(default_factory=tuple)
    c_terms: Tuple[CollapseTerm, ...] = field(default_factory=tuple)

    observables: Optional[Observables] = None

    args: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


DriveKind = Literal["1ph", "2ph", "unresolved"]


@dataclass(frozen=True)
class ResolvedDrive:
    drive_id: str
    physical: ClassicalCoherentDrive
    kind: DriveKind
    # optional coherent combination
    components: Tuple[Tuple[Transition, complex], ...]
    transition: Optional[Transition]  # if single component
    # or store omega_tr and compute later
    detuning: Callable[[float], float] | float | None
    candidates: Tuple[Transition, ...]
    meta: Dict[str, Any]


@dataclass(frozen=True)
class DriveCoefficients:
    omega_by_transition: Dict[Transition, CoeffExpr]
    omega_2ph: Optional[CoeffExpr] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RatesBundle:
    """
    Collection of rates used to bind CollapseTerms and sometimes Hamiltonian renorm.

    rates maps labels -> CoeffExpr (ConstCoeff or time-dependent expression).
    args is a dict forwarded to solver coefficient callables (QuTiP args).
    """

    rates: Dict[str, CoeffExpr] = field(default_factory=dict)
    args: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Optional[CoeffExpr]:
        return self.rates.get(key)


@dataclass(frozen=True)
class TruncationPolicy:
    pol_dim: int = 2
    per_pol_dim: Optional[Callable[[int, str], int]] = None

    def dim_for(self, mode_index: int, pol: str) -> int:
        if self.per_pol_dim is None:
            return int(self.pol_dim)
        return int(self.per_pol_dim(mode_index, pol))


def build_dims(qd, trunc: TruncationPolicy) -> List[int]:
    dims: List[int] = [4]
    for i, _m in enumerate(qd.modes.modes):
        dims.append(trunc.dim_for(i, "+"))
        dims.append(trunc.dim_for(i, "-"))
    return dims


@dataclass(frozen=True)
class MESimulationResult:
    tlist: np.ndarray
    expect: Dict[str, np.ndarray]
    states: Optional[Tuple[np.ndarray, ...]]
    final_state: Optional[np.ndarray]
    meta: Dict[str, Any]
