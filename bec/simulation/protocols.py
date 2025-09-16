from __future__ import annotations
from typing import Protocol, Optional, Sequence, List, Dict, Any, TYPE_CHECKING
import numpy as np
from qutip import Qobj
from bec.light.classical import ClassicalTwoPhotonDrive
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.composite_envelope import CompositeEnvelope
from bec.quantum_dot import QuantumDot

if TYPE_CHECKING:
    # QuTiP’s result object; if the import path changes in your version,
    # switch this to the correct one or just use `Any`.
    # type: ignore[attr-defined]
    from qutip.solver import Result as QutipResult
else:
    QutipResult = Any  # graceful fallback for runtime


class Scenario(Protocol):
    """
    Provides/installs external modes and (optionally) a classical 2g drive.
    """

    def prepare(self, qd: QuantumDot) -> None: ...
    def classical_drive(self) -> Optional[ClassicalTwoPhotonDrive]: ...
    def label(self) -> str: ...


class SpaceBuilder(Protocol):
    """
    Build composite space (QD + Fock), bind mode containers, and return rho0.
    """

    def build_space(
        self, qd: QuantumDot, trunc_per_pol: int
    ) -> tuple[list[Envelope], list[Fock], CompositeEnvelope]: ...

    def build_qutip_space(
        self, cstate: CompositeEnvelope, qd_dot, focks: list[Fock]
    ) -> tuple[list[int], list[list[int]], Qobj]: ...


class HamiltonianComposer(Protocol):
    """Produce QuTiP H-list from qd + drive + dims + tlist."""

    def compose(
        self,
        qd: QuantumDot,
        dims: list[int],
        drive: Optional[ClassicalTwoPhotonDrive],
        time_unit_s: float,
    ) -> list: ...


class CollapseComposer(Protocol):
    """Build collapse operators."""

    def compose(
        self, qd: QuantumDot, dims: list[int], time_unit_s: float
    ) -> list[Qobj]: ...


class ObservableComposer(Protocol):
    """Build projectors/observables and return (qd_proj, lm_proj) dicts."""

    def compose_qd(
        self, qd: QuantumDot, dims: list[int], time_unit_s: float
    ) -> Dict[str, Qobj]: ...

    def compose_lm(
        self, qd: QuantumDot, dims: list[int], time_unit_s: float
    ) -> Dict[str, Qobj]: ...


class ExpectationLayout(Protocol):
    """Select e_ops list and define index slices to unpack expectations."""

    def select(
        self, qd_proj: Dict[str, Qobj], lm_proj: Dict[str, Qobj], qd: QuantumDot
    ) -> tuple[list[Qobj], Dict[str, slice]]: ...


class SolverBackend(Protocol):
    """Run QuTiP mesolve with standard options."""

    def solve(
        self,
        H: list,
        rho0: Qobj,
        tlist: np.ndarray,
        c_ops: list[Qobj],
        e_ops: list[Qobj],
    ) -> QutipResult: ...
