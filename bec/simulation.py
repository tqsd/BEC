from dataclasses import dataclass
from typing import Optional, Sequence, Union, List, Dict
import numpy as np
from qutip import Qobj, mesolve, Options

from photon_weave.operation import Operation
from photon_weave.state.custom_state import CustomState
from photon_weave.state.fock import Fock
from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.polarization import PolarizationLabel

from bec.light.classical import ClassicalTwoPhotonDrive
from bec.quantum_dot.qd_photon_weave import QuantumDotSystem


# ---------------- data container ----------------


@dataclass
class QDTraces:
    t: np.ndarray
    classical: bool
    flying_labels: List[str]
    intrinsic_labels: List[str]
    qd: List[np.ndarray]  # [|G>, |X1>, |X2>, |XX>]
    fly_H: List[np.ndarray]
    fly_V: List[np.ndarray]
    out_H: List[np.ndarray]
    out_V: List[np.ndarray]
    omega: Optional[np.ndarray] = None  # if classical drive used
    # cumulative area if classical drive used
    area: Optional[np.ndarray] = None


# ---------------- simulation only ----------------


class QDSimulator:
    """
    Builds the composite space, Hamiltonians, collapse ops, runs QuTiP,
    and returns QDTraces (no plotting).
    """

    def __init__(
        self,
        qd: QuantumDotSystem,
        classical_2g: Optional[ClassicalTwoPhotonDrive] = None,
        tlist: Optional[Union[Sequence[float], np.ndarray]] = None,
        trunc_per_pol: int = 2,
    ):
        self.qd = qd
        self.classical_2g = classical_2g
        self.tlist = (
            np.asarray(tlist)
            if tlist is not None
            else np.linspace(0.0, 10.0, 500)
        )
        self.trunc_per_pol = int(trunc_per_pol)

        # Will be filled during build steps
        self.ENVS: list[Envelope] = []
        self.FOCKS: list[Fock] = []
        self.CSTATE: Optional[CompositeEnvelope] = None
        self.DIMENSIONS: list[int] = []
        self.DIMS: list[list[int]] = []
        self.rho0: Optional[Qobj] = None
        self.H: Optional[list] = None
        self.C_OPS: Optional[list[Qobj]] = None
        self.P_ops: Dict[str, Qobj] = {}
        self.LM_ops: Dict[str, Qobj] = {}

        # indices for expectation unpacking
        self.idx_qd = slice(0, 0)
        self.idx_fly_T = slice(0, 0)
        self.idx_fly_H = slice(0, 0)
        self.idx_fly_V = slice(0, 0)
        self.idx_out_T = slice(0, 0)
        self.idx_out_H = slice(0, 0)
        self.idx_out_V = slice(0, 0)

        # labels (filled later)
        self.flying_labels: List[str] = []
        self.intrinsic_labels: List[str] = []

        self._build_space()

    # ---------- public API ----------

    def apply_operation(
        self,
        op: Operation,
        *labels: Union[str, CustomState],
        pol: Optional[str] = None,
    ) -> None:
        """
        Apply `op` to the states addressed by each label.
        """
        if self.CSTATE is None:
            raise ValueError("Composite space not built yet.")

        def _targets_for_label(
            lbl: Union[str, CustomState],
        ) -> list[Union[Fock, CustomState]]:
            if isinstance(lbl, CustomState):
                return [lbl]

            key = lbl.lower()
            if key in ("qd", "dot"):
                return [self.qd.dot]

            mode = self.qd.filter_modes(lbl)  # raises if not found
            if not hasattr(mode, "containerHV") or mode.containerHV is None:
                raise ValueError(
                    f"Mode '{
                        lbl}' has no containerHV. Ensure _build_space() has run."
                )
            env_h, env_v = mode.containerHV

            if pol is None:
                return [env_h.fock, env_v.fock]
            p = pol.upper()
            if p == "H":
                return [env_h.fock]
            if p == "V":
                return [env_v.fock]
            raise ValueError("pol must be None, 'H', or 'V'")

        targets = []
        for lbl in labels:
            targets.extend(_targets_for_label(lbl))
        if not targets:
            print("No operation applied")
            return
        self.CSTATE.apply_operation(op, *targets)
        self.rho0 = None  # invalidate if already built

    def compute_traces(self, compute_top: bool = True) -> QDTraces:
        """Full build → solve → extract expectation values into QDTraces."""
        self._build_qutip_space()
        self._build_hamiltonian_and_collapse()
        self._build_operators()
        e_ops_list = self._make_eops()
        result = self._solve(e_ops_list)

        # extract expectations
        qd_traces = result.expect[self.idx_qd]
        fly_H = result.expect[self.idx_fly_H]
        fly_V = result.expect[self.idx_fly_V]
        out_H = result.expect[self.idx_out_H]
        out_V = result.expect[self.idx_out_V]

        # optional classical panel signals
        Omega_t = area_t = None
        if compute_top and (self.classical_2g is not None):
            Omega_fn = self.classical_2g.qutip_coeff()
            Omega_t = np.array([Omega_fn(t, {}) for t in self.tlist])
            area_t = np.concatenate(
                [
                    [0.0],
                    np.cumsum(
                        0.5 * np.diff(self.tlist) *
                        (Omega_t[1:] + Omega_t[:-1])
                    ),
                ]
            )

        return QDTraces(
            t=self.tlist,
            classical=(self.classical_2g is not None),
            flying_labels=self.flying_labels,
            intrinsic_labels=self.intrinsic_labels,
            qd=[qd_traces[i] for i in range(4)],
            fly_H=list(fly_H),
            fly_V=list(fly_V),
            out_H=list(out_H),
            out_V=list(out_V),
            omega=Omega_t,
            area=area_t,
        )

    # ---------- build steps (private) ----------

    def _build_space(self):
        self.ENVS = []
        for mode in self.qd.modes:
            env_h = Envelope()
            env_h.fock.dimensions = self.trunc_per_pol
            env_v = Envelope()
            env_v.polarization.state = PolarizationLabel.V
            env_v.fock.dimensions = self.trunc_per_pol
            mode.containerHV = [env_h, env_v]
            self.ENVS.extend([env_h, env_v])

        self.CSTATE = CompositeEnvelope(self.qd.dot, *self.ENVS)
        self.FOCKS = [env.fock for env in self.ENVS]
        self.CSTATE.combine(self.qd.dot, *self.FOCKS)
        self.CSTATE.reorder(self.qd.dot, *self.FOCKS)

    def _build_qutip_space(self):
        self.CSTATE.reorder(self.qd.dot, *self.FOCKS)
        self.qd.dot.expand()
        self.DIMENSIONS = [s.dimensions for s in [self.qd.dot, *self.FOCKS]]
        self.DIMS = [self.DIMENSIONS, self.DIMENSIONS]
        self.rho0 = Qobj(
            np.array(self.CSTATE.product_states[0].state), dims=self.DIMS
        ).to("csr")

    def _build_hamiltonian_and_collapse(self):
        self.H = self.qd.build_hamiltonians(
            dims=self.DIMENSIONS,
            classical_2g=self.classical_2g,
            tlist=self.tlist,
        )
        self.C_OPS = self.qd.qutip_collapse_operators(self.DIMENSIONS)

    def _build_operators(self):
        self.P_ops = self.qd.qutip_projectors(self.DIMENSIONS)
        self.LM_ops = self.qd.qutip_light_mode_projectors(self.DIMENSIONS)

    def _make_eops(self):
        # QD pops
        qd_eops = [self.P_ops[k] for k in ("P_G", "P_X1", "P_X2", "P_XX")]

        # Labels
        self.flying_labels = [
            m.label
            for m in self.qd.modes
            if getattr(m, "source", None) == "external"
        ]
        self.intrinsic_labels = [
            m.label
            for m in self.qd.modes
            if getattr(m, "source", None) == "internal"
        ]

        # Flying (inputs)
        fly_T = [
            self.LM_ops[f"N[{lbl}]"]
            for lbl in self.flying_labels
            if f"N[{lbl}]" in self.LM_ops
        ]
        fly_H = [
            self.LM_ops[f"N-[{lbl}]"]
            for lbl in self.flying_labels
            if f"N-[{lbl}]" in self.LM_ops
        ]
        fly_V = [
            self.LM_ops[f"N+[{lbl}]"]
            for lbl in self.flying_labels
            if f"N+[{lbl}]" in self.LM_ops
        ]

        # Intrinsic (outputs)
        out_T = [
            self.LM_ops[f"N[{lbl}]"]
            for lbl in self.intrinsic_labels
            if f"N[{lbl}]" in self.LM_ops
        ]
        out_H = [
            self.LM_ops[f"N-[{lbl}]"]
            for lbl in self.intrinsic_labels
            if f"N-[{lbl}]" in self.LM_ops
        ]
        out_V = [
            self.LM_ops[f"N+[{lbl}]"]
            for lbl in self.intrinsic_labels
            if f"N+[{lbl}]" in self.LM_ops
        ]

        e_ops_list = [
            op.to("csr")
            for op in (qd_eops + fly_T + fly_H + fly_V + out_T + out_H + out_V)
        ]

        # store indices
        i0 = 0
        self.idx_qd = slice(i0, i0 + len(qd_eops))
        i0 += len(qd_eops)
        self.idx_fly_T = slice(i0, i0 + len(fly_T))
        i0 += len(fly_T)
        self.idx_fly_H = slice(i0, i0 + len(fly_H))
        i0 += len(fly_H)
        self.idx_fly_V = slice(i0, i0 + len(fly_V))
        i0 += len(fly_V)
        self.idx_out_T = slice(i0, i0 + len(out_T))
        i0 += len(out_T)
        self.idx_out_H = slice(i0, i0 + len(out_H))
        i0 += len(out_H)
        self.idx_out_V = slice(i0, i0 + len(out_V))
        i0 += len(out_V)

        return e_ops_list

    def _solve(self, e_ops_list):
        opts = Options(nsteps=10000, rtol=1e-9,
                       atol=1e-11, progress_bar="tqdm")
        return mesolve(
            self.H,
            self.rho0,
            self.tlist,
            c_ops=self.C_OPS,
            e_ops=e_ops_list,
            options=opts,
        )
