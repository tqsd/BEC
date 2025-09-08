from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union, List
from photon_weave.operation import Operation
from photon_weave.state.custom_state import CustomState
from photon_weave.state.fock import Fock
from qutip import Qobj, mesolve, Options

from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.polarization import PolarizationLabel

from bec.light.classical import ClassicalTwoPhotonDrive
from bec.quantum_dot.qd_photon_weave import QuantumDotSystem


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
    omega: Optional[np.ndarray] = None  # for classical drive
    area: Optional[np.ndarray] = None  # for classical drive


class QDPlotter:
    """
    One-shot runner/plotter for QuantumDotSystem simulations.

    Usage:
        plotter = QDPlotter(QD, classical_2g=DRIVE, tlist=np.linspace(0,10,500))
        result = plotter.run_and_plot(filename="biexciton_column", show_top=True)
    """

    def __init__(
        self,
        qd: QuantumDotSystem,
        classical_2g: Optional[ClassicalTwoPhotonDrive] = None,
        tlist: Optional[Union[Sequence[float], np.ndarray]] = None,
        trunc_per_pol: int = 2,
        time_label: str = "Time (ns)",
    ):
        self.qd = qd
        self.classical_2g = classical_2g
        self.tlist = (
            np.asarray(tlist)
            if tlist is not None
            else np.linspace(0.0, 10.0, 500)
        )
        self.trunc_per_pol = int(trunc_per_pol)
        self.time_label = time_label

        # Will be filled during build steps
        self.ENVS: list[Envelope] = []
        self.DIMENSIONS: list[int] = []
        self.DIMS: list[list[int]] = []
        self.CSTATE: Optional[CompositeEnvelope] = None
        self.rho0: Optional[Qobj] = None
        self.H: Optional[list] = None
        self.C_OPS: Optional[list[Qobj]] = None
        self.P_ops: dict[str, Qobj] = {}
        self.LM_ops: dict[str, Qobj] = {}

        # indices for expectation unpacking
        self.idx_qd = slice(0, 0)
        self.idx_fly_T = slice(0, 0)
        self.idx_fly_H = slice(0, 0)
        self.idx_fly_V = slice(0, 0)
        self.idx_out_T = slice(0, 0)
        self.idx_out_H = slice(0, 0)
        self.idx_out_V = slice(0, 0)

        # build the space
        self._build_space()

    # ---------- build the composite space ----------
    def _build_space(self):
        self.ENVS = []

        # 1) Build two envelopes per optical mode (H/V)
        for mode in self.qd.modes:
            env_h = Envelope()
            env_h.fock.dimensions = self.trunc_per_pol  # H

            env_v = Envelope()
            env_v.polarization.state = PolarizationLabel.V
            env_v.fock.dimensions = self.trunc_per_pol  # V

            mode.containerHV = [env_h, env_v]
            self.ENVS.extend([env_h, env_v])

        # 2) Construct the composite with the envelopes (not strictly needed for reorder)
        self.CSTATE = CompositeEnvelope(self.qd.dot, *self.ENVS)

        # 3) IMPORTANT: capture fock objects ONCE and reuse
        self.FOCKS = [env.fock for env in self.ENVS]

        # 4) Build product basis over QD + FOCKS
        self.CSTATE.combine(self.qd.dot, *self.FOCKS)

        # 5) Reorder BEFORE expanding the dot (object identities still match)
        self.CSTATE.reorder(self.qd.dot, *self.FOCKS)

    def _build_qutip_space(self):
        self.CSTATE.reorder(self.qd.dot, *self.FOCKS)

        # 6) Now expand QD and compute dims
        self.qd.dot.expand()

        self.DIMENSIONS = [s.dimensions for s in [self.qd.dot, *self.FOCKS]]
        self.DIMS = [self.DIMENSIONS, self.DIMENSIONS]

        # 7) Initial state
        self.rho0 = Qobj(
            np.array(self.CSTATE.product_states[0].state), dims=self.DIMS
        ).to("csr")

    # ---------- Hamiltonians & collapse ----------

    def _build_hamiltonian_and_collapse(self):
        self.H = self.qd.build_hamiltonians(
            dims=self.DIMENSIONS,
            classical_2g=self.classical_2g,
            tlist=self.tlist,
        )
        self.C_OPS = self.qd.qutip_collapse_operators(self.DIMENSIONS)

    # ---------- projectors (QD + light modes) ----------
    def _build_operators(self):
        self.P_ops = self.qd.qutip_projectors(self.DIMENSIONS)
        self.LM_ops = self.qd.qutip_light_mode_projectors(self.DIMENSIONS)

    # ---------- assemble e_ops and indices ----------
    def _make_eops(self):
        # QD populations
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

    # ---------- solve ----------
    def _solve(self, e_ops_list):
        opts = Options(nsteps=10000, rtol=1e-9, atol=1e-11, progress_bar="tqdm")
        return mesolve(
            self.H,
            self.rho0,
            self.tlist,
            c_ops=self.C_OPS,
            e_ops=e_ops_list,
            options=opts,
        )

    # ---------- plotting ----------
    def _plot(self, result, filename: Optional[str], show_top: bool):
        # unpack expectations
        qd_traces = result.expect[self.idx_qd]
        fly_T_traces = result.expect[self.idx_fly_T]
        fly_H_traces = result.expect[self.idx_fly_H]
        fly_V_traces = result.expect[self.idx_fly_V]
        out_T_traces = result.expect[self.idx_out_T]
        out_H_traces = result.expect[self.idx_out_H]
        out_V_traces = result.expect[self.idx_out_V]

        # decide how many rows: with top panel or not
        nrows = 3 if show_top else 2
        fig_axes = plt.subplots(nrows, 1, figsize=(7, 2.2 * nrows), sharex=True)
        if nrows == 3:
            fig, (ax_top, ax_mid, ax_bot) = fig_axes
        else:
            fig, (ax_mid, ax_bot) = fig_axes
            ax_top = None

        # ----- TOP panel -----
        if show_top:
            if self.classical_2g is not None:
                # plot Omega(t) and pulse area
                Omega_fn = self.classical_2g.qutip_coeff()
                Omega_t = np.array([Omega_fn(t, {}) for t in self.tlist])
                area_t = np.concatenate(
                    [
                        [0.0],
                        np.cumsum(
                            0.5
                            * np.diff(self.tlist)
                            * (Omega_t[1:] + Omega_t[:-1])
                        ),
                    ]
                )
                ax_top.plot(self.tlist, Omega_t, label=r"$\Omega(t)$")
                ax_top_t = ax_top.twinx()
                ax_top_t.plot(
                    self.tlist, area_t, ls="--", alpha=0.85, label="area"
                )
                ax_top.set_ylabel(r"$\Omega(t)$")
                ax_top_t.set_ylabel(r"$\int^t \Omega(t^\prime)\,dt^\prime$")
                ax_top.grid(True, alpha=0.3)
            else:
                # plot flying mode photon numbers (H and V)
                print("PLOTTING QUANTUM")
                for k, lbl in enumerate(self.flying_labels):
                    print("PLOTTING", lbl)
                    if len(fly_H_traces) > k:
                        ax_top.plot(
                            self.tlist,
                            fly_H_traces[k],
                            label=f"{lbl} (H)",
                            alpha=0.7,
                        )
                    if len(fly_V_traces) > k:
                        ax_top.plot(
                            self.tlist,
                            fly_V_traces[k],
                            ls="--",
                            label=f"{lbl} (V)",
                            alpha=0.7,
                        )
                ax_top.set_ylabel(r"$\langle N\rangle$ (in)")
                if self.flying_labels:
                    ax_top.legend(loc="best", fontsize=9, ncol=2)
                ax_top.grid(True, alpha=0.3)

        # ----- MID: QD populations -----
        labels_qd = [
            r"$|G\rangle$",
            r"$|X_1\rangle$",
            r"$|X_2\rangle$",
            r"$|XX\rangle$",
        ]
        for lab, y in zip(labels_qd, qd_traces):
            ax_mid.plot(self.tlist, y, label=lab)
        ax_mid.set_ylabel("QD")
        ax_mid.legend(loc="best", ncol=2, fontsize=9)
        ax_mid.grid(True, alpha=0.3)

        # ----- BOT: intrinsic output photons -----
        for k, lbl in enumerate(self.intrinsic_labels):
            if len(out_H_traces) > k:
                ax_bot.plot(self.tlist, out_H_traces[k], label=f"{lbl} (H)")
            if len(out_V_traces) > k:
                ax_bot.plot(
                    self.tlist, out_V_traces[k], ls="--", label=f"{lbl} (V)"
                )
        ax_bot.set_xlabel(self.time_label)
        ax_bot.set_ylabel(r"$\langle N\rangle$ (out)")
        if self.intrinsic_labels:
            ax_bot.legend(loc="best", fontsize=9, ncol=2)
        ax_bot.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if requested
        if filename:
            if filename.lower().endswith((".png", ".pdf")):
                fig.savefig(
                    filename,
                    dpi=300 if filename.endswith(".png") else None,
                    bbox_inches="tight",
                )
            else:
                fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
                fig.savefig(f"{filename}.pdf", bbox_inches="tight")
        return fig

    # ---------- public API ----------
    def run_and_plot(
        self, filename: Optional[str] = None, show_top: bool = True
    ):
        self._build_qutip_space()
        self._build_hamiltonian_and_collapse()
        self._build_operators()
        e_ops_list = self._make_eops()
        result = self._solve(e_ops_list)
        fig = self._plot(result, filename=filename, show_top=show_top)
        return result

    def apply_operation(
        self,
        op: Operation,
        *labels: Union[str, CustomState],
        pol: Optional[str] = None,
    ) -> None:
        """
        Apply `op` to the states addressed by each label.

        Labels can be:
        - a mode label string (e.g. "second") → applies to that mode's Fock spaces
            (by default both H and V, or only one if `pol="H"` / `pol="V"`).
        - "qd" or "dot" → applies to the QD CustomState
        - a CustomState instance (advanced)

        Notes:
        - Call this BEFORE run_and_plot() so it modifies the initial product state.
        - The order of Fock states is [H, V] when both are used.
        """
        if self.CSTATE is None:
            raise ValueError("Composite space not built yet.")

        def _targets_for_label(
            lbl: Union[str, CustomState],
        ) -> list[Union[Fock, CustomState]]:
            # Direct CustomState
            if isinstance(lbl, CustomState):
                return [lbl]

            # String labels
            key = lbl.lower()
            if key in ("qd", "dot"):
                return [self.qd.dot]

            # Optical mode
            mode = self.qd.filter_modes(lbl)  # raises if not found
            if not hasattr(mode, "containerHV") or mode.containerHV is None:
                raise ValueError(
                    f"Mode '{
                        lbl}' has no containerHV. Make sure QDPlotter._build_space() has run."
                )
            env_h, env_v = mode.containerHV  # set in _build_space()

            if pol is None:
                return [env_h.fock, env_v.fock]  # default: H then V
            p = pol.upper()
            if p == "H":
                return [env_h.fock]
            if p == "V":
                return [env_v.fock]
            raise ValueError("pol must be None, 'H', or 'V'")

        # Apply op for each label independently
        targets = []
        for lbl in labels:
            targets.extend(_targets_for_label(lbl))
        if len(targets) == 0:
            print("No opeation is applied")
        self.CSTATE.apply_operation(op, *targets)

        # Invalidate rho0 if we already built the QuTiP objects
        self.rho0 = None

    def compute_traces(self, show_top: bool = True) -> QDTraces:
        # build & solve (same steps as run_and_plot but without plotting)
        self._build_qutip_space()
        self._build_hamiltonian_and_collapse()
        self._build_operators()
        e_ops_list = self._make_eops()
        result = self._solve(e_ops_list)

        # extract expectations using your stored slices
        qd_traces = result.expect[self.idx_qd]
        fly_H = result.expect[self.idx_fly_H]
        fly_V = result.expect[self.idx_fly_V]
        out_H = result.expect[self.idx_out_H]
        out_V = result.expect[self.idx_out_V]

        # optional top-panel info for classical drive
        Omega_t = area_t = None
        if show_top and (self.classical_2g is not None):
            Omega_fn = self.classical_2g.qutip_coeff()
            Omega_t = np.array([Omega_fn(t, {}) for t in self.tlist])
            area_t = np.concatenate(
                [
                    [0.0],
                    np.cumsum(
                        0.5 * np.diff(self.tlist) * (Omega_t[1:] + Omega_t[:-1])
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
