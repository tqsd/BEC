from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from bec.quantum_dot.dot import QuantumDot
from bec.light.classical import ClassicalFieldDrive

from bec.simulation.types import (
    DriveCoefficients,
    MEProblem,
    RatesBundle,
    ResolvedDrive,
    MESimulationResult,
)

# ----------------------
# Compile configurations
# ----------------------


@dataclass(frozen=True)
class CompileConfig:
    """
    Configuration used during the compilation

    tlist is in solver units
    time_unit_s is seconds per solver unit
    """

    tlist: np.ndarray
    trunc_per_pol: int = 2
    time_unit_s: float = 1.0


# ------------------
# Scenario Interface
# ------------------


@runtime_checkable
class Scenario(Protocol):
    """
    Scenario prepares the dot (registers modes, etc.) and provides drives.
    """

    def prepare(self, qd: QuantumDot) -> None: ...
    def drives(self) -> Sequence[ClassicalFieldDrive]: ...


@runtime_checkable
class DriveDecoder(Protocol):
    """
    Decodes raw drives into resolved 'intent' objects
    """

    def decode(
        self,
        qd: QuantumDot,
        drives: Sequence[ClassicalFieldDrive],
        *,
        time_unit_s: float,
        tlist: np.ndarray,
    ) -> Tuple[ResolvedDrive, ...]: ...


class DriveStrengthModel(Protocol):
    r"""
    Builds solver-ready \Omega(t) coefficient expressions for a decoded drive.

    - Does NOT decide which transition is driven (decoder already did).
    - Produces CoeffExpr callables in solver units (t is solver time).
    """

    def build(
        self, *, qd: QuantumDot, rd: ResolvedDrive, time_unit_s: float
    ) -> DriveCoefficients: ...

    def build_many(
        self,
        *,
        qd: QuantumDot,
        resolved: tuple[ResolvedDrive, ...],
        time_unit_s: float,
    ) -> Dict[str, DriveCoefficients]: ...


@runtime_checkable
class HamiltonianComposer(Protocol):
    def compose(
        self,
        *,
        qd: QuantumDot,
        dims: List[int],
        resolved: Tuple[ResolvedDrive, ...],
        drive_coeffs: Dict[str, DriveCoefficients],
        time_unit_s: float,
        tlist: Optional[np.ndarray] = None,
    ) -> List[Any]: ...


@runtime_checkable
class RatesStage(Protocol):
    """
    Computes model rates for radiative / phonon channels
    """

    def compute(
        self,
        *,
        qd: QuantumDot,
        resolved: Tuple[ResolvedDrive, ...],
        drive_coeffs: Optional[Dict[str, DriveCoefficients]] = None,
        time_unit_s: float,
        tlist: np.ndarray,
    ) -> RatesBundle: ...


@runtime_checkable
class CollapseComposer(Protocol):
    """
    Produces collapse IR terms from computed rates and resolved drives.
    """

    def compose(
        self,
        *,
        qd: QuantumDot,
        dims: List[int],
        rates: RatesBundle,
        time_unit_s: float,
        tlist: Optional[np.ndarray] = None,
    ) -> List[Any]: ...


@runtime_checkable
class ObservablesComposer(Protocol):
    """
    Produces observables bundle.
    """

    def compose(self, qd: QuantumDot, dims: List[int]) -> Any: ...


# ----------------------------
# Compiler facade protocol
# ----------------------------


@runtime_checkable
class ProblemCompiler(Protocol):
    """
    Facade that compiles QD + scenario + config into MEProblem.
    """

    def compile(
        self, qd: QuantumDot, scenario: Scenario, cfg: CompileConfig
    ) -> MEProblem: ...


# ----------------------------
# Solver backend protocol
# ----------------------------
@runtime_checkable
class SolverBackend(Protocol):
    """
    Consumes MEProblem and returns a backend-specific result.
    """

    def solve(self, problem: MEProblem) -> Any: ...


@runtime_checkable
class SimulationAdapter(Protocol):
    """
    Backend Plugin: consumes a backend-agnostic MEProblem and returns
    a backend-agnostic MESimulationResult.
    """

    backend_name: str

    def simulate(
        self,
        me: MEProblem,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> MESimulationResult: ...


TransitionKey = Hashable


class TransitionRegistryView(Protocol):
    """QD-free view of available optical transitions."""

    def transitions(self) -> Tuple[TransitionKey, ...]: ...

    def kind(self, tr: TransitionKey) -> str:
        """Return '1ph' or '2ph' (or other strings if you extend later)."""
        ...

    def omega_ref_rad_s(self, tr: TransitionKey) -> float:
        """Physical angular frequency ω_tr in rad/s."""
        ...


class PolarizationCouplingView(Protocol):
    """
    Maps drive polarization into coupling weights for transitions.
    Must be QD-aware internally, but interface is QD-free.
    """

    def coupling_weight(self, tr: TransitionKey, E: np.ndarray) -> complex:
        """
        Return complex overlap c = d* · E_eff for that transition.
        If unknown/unconstrained, return 1+0j.
        """
        ...


class BandwidthEstimator(Protocol):
    def sigma_omega_rad_s(
        self,
        *,
        drive: Any,
        tlist_solver: np.ndarray,
        time_unit_s: float,
    ) -> float: ...


@dataclass(frozen=True)
class DriveDecodeContext:
    transitions: TransitionRegistryView
    pol: Optional[PolarizationCouplingView] = None
    bandwidth: Optional[BandwidthEstimator] = None
