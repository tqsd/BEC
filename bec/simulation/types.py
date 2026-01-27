from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
)

import numpy as np

from bec.simulation.coeffs import CoeffExpr


# -------------------------
# Opaque keys / IDs
# -------------------------

TransitionKey = Hashable  # e.g. "G_X1", enum value, tuple, etc.
ChannelKey = Hashable  # e.g. ("lambda0", "H"), or "ch0_plus", etc.
DriveId = str
TermKind = Literal["H", "C", "E"]


# -------------------------
# Minimal “compiled model” protocol
# -------------------------


class CompiledModel(Protocol):
    """
    Anything definition-layer can pass through MEProblem without coupling.
    The runtime doesn't need to know what it is; it's metadata/handle only.
    """

    def model_id(self) -> str: ...


# -------------------------
# Runtime term types
# -------------------------


@dataclass(frozen=True)
class HamiltonianTerm:
    label: str
    op: np.ndarray  # numeric, unitless
    coeff: Optional[CoeffExpr] = None  # None => static
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CollapseTerm:
    label: str
    op: np.ndarray  # numeric, unitless
    coeff: CoeffExpr  # must exist
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observables:
    """
    Numeric, unitless observables. Keep separate dicts if you like that structure,
    but one dict is often enough.
    """

    ops: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, label: str, op: np.ndarray, **meta: Any) -> None:
        # Not mutating because frozen; convenience would be builder-side.
        raise TypeError(
            "Observables is frozen; build via a builder or use a mutable builder class."
        )


# -------------------------
# Drives (runtime-side view)
# -------------------------

DriveKind = Literal["1ph", "2ph", "unresolved"]


@dataclass(frozen=True)
class ResolvedDrive:
    """
    Runtime-side drive resolution output.

    This must not depend on QuantumDot/Transition enums. Use TransitionKey.
    """

    drive_id: DriveId
    kind: DriveKind

    # “physical” object is opaque: can be your ClassicalCoherentDrive, or any drive object.
    physical: Any

    # coherent combination (transition, complex weight)
    components: Tuple[Tuple[TransitionKey, complex], ...] = ()
    transition: Optional[TransitionKey] = None

    # detuning in solver-time units; can be constant (float) or callable (t->float)
    detuning: Callable[[float], float] | float | None = None

    candidates: Tuple[TransitionKey, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriveCoefficients:
    """
    Bound coefficients in solver-time units.
    """

    omega_by_transition: Dict[TransitionKey, CoeffExpr] = field(
        default_factory=dict
    )
    omega_2ph: Optional[CoeffExpr] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# -------------------------
# Rates bundle
# -------------------------

Args = Mapping[str, Any]


@dataclass(frozen=True)
class RatesBundle:
    rates: Dict[str, CoeffExpr] = field(default_factory=dict)
    args: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Optional[CoeffExpr]:
        return self.rates.get(key)


# -------------------------
# Truncation policy + dims building
# -------------------------


@dataclass(frozen=True)
class TruncationPolicy:
    pol_dim: int = 2
    per_pol_dim: Optional[Callable[[int, str], int]] = None

    def dim_for(self, mode_index: int, pol: str) -> int:
        if self.per_pol_dim is None:
            return int(self.pol_dim)
        return int(self.per_pol_dim(mode_index, pol))


class ModeRegistryView(Protocol):
    """
    Minimal view the runtime compiler needs to build dims, without importing QD code.

    Your QD can expose an adapter that implements this view.
    """

    def num_modes(self) -> int: ...


def build_dims(
    mode_registry: ModeRegistryView, trunc: TruncationPolicy, *, qd_dim: int = 4
) -> List[int]:
    dims: List[int] = [int(qd_dim)]
    nm = int(len(mode_registry.channels))
    for i in range(nm):
        dims.append(trunc.dim_for(i, "+"))
        dims.append(trunc.dim_for(i, "-"))
    return dims


# -------------------------
# MEProblem + result
# -------------------------


@dataclass(frozen=True)
class MEProblem:
    """
    Backend-agnostic, solver-ready ME container.

    IMPORTANT: units are already stripped. Solver time is dimensionless,
    with time_unit_s mapping solver-time -> seconds.
    """

    model: Any  # optional handle (QuantumDot or metadata); keep opaque
    tlist: np.ndarray  # solver-time axis
    time_unit_s: float  # seconds per solver-time unit

    dims: List[int]
    rho0: np.ndarray

    h_terms: Tuple[HamiltonianTerm, ...] = field(default_factory=tuple)
    c_terms: Tuple[CollapseTerm, ...] = field(default_factory=tuple)
    observables: Optional[Observables] = None

    args: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def t_phys(self) -> np.ndarray:
        """Physical seconds for plotting/debug."""
        return np.asarray(self.tlist, dtype=float) * float(self.time_unit_s)


@dataclass(frozen=True)
class MESimulationResult:
    tlist: np.ndarray
    expect: Dict[str, np.ndarray] = field(default_factory=dict)
    states: Optional[Tuple[np.ndarray, ...]] = None
    final_state: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)
