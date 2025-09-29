# bec/simulation/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from qutip import Qobj
from bec.light.detuning import two_photon_detuning_profile
from bec.params.transitions import TransitionType
from bec.quantum_dot import QuantumDot
from bec.simulation.qd_traces import QDTraces
from bec.simulation.collapse_composer import DefaultCollapseComposer
from bec.simulation.expectation_layouts import DefaultExpectationLayout
from bec.simulation.hamiltonian_composers import DefaultHamiltonianComposer
from bec.simulation.observable_composer import DefaultObservableComposer
from bec.simulation.solvers import QutipMesolveBackend
from bec.simulation.space_builder import (
    DefaultSpaceBuilder,
)  # only for examples/types

from .protocols import (
    Scenario,
    SpaceBuilder,
    HamiltonianComposer,
    CollapseComposer,
    ObservableComposer,
    ExpectationLayout,
    SolverBackend,
)


@dataclass
class SimulationConfig:
    tlist: np.ndarray
    trunc_per_pol: int = 2
    time_unit_s: float = 1.0


class SimulationEngine:
    """
    Orchestrates:
      - scenario.prepare(qd)
      - build space → rho0
      - H, c_ops, e_ops
      - mesolve
      - pack results as QDTraces (same shape you already use)
    """

    def __init__(
        self,
        space: SpaceBuilder | None = None,
        hams: HamiltonianComposer | None = None,
        collapses: CollapseComposer | None = None,
        observables: ObservableComposer | None = None,
        layout: ExpectationLayout | None = None,
        solver: SolverBackend | None = None,
    ):
        self.space = space or DefaultSpaceBuilder()
        self.hams = hams or DefaultHamiltonianComposer()
        self.collapses = collapses or DefaultCollapseComposer()
        self.observables = observables or DefaultObservableComposer()
        self.layout = layout or DefaultExpectationLayout()
        self.solver = solver or QutipMesolveBackend()

    def run_with_state(
        self,
        qd: QuantumDot,
        scenario: Scenario,
        cfg: SimulationConfig,
        reduce_photonic: bool = True,
    ) -> Tuple[QDTraces, Optional[Qobj], Optional[Qobj]]:
        """
        Like `run`, but also returns:
        - rho_final: the final composite density matrix (QD ⊗ all Fock)
        - rho_phot_final: the reduced photonic state with QD traced out (if requested)
        """
        # 1) install modes / classical drive
        qd.N_cut = cfg.trunc_per_pol
        scenario.prepare(qd)
        drive = scenario.classical_drive()

        # 2) build space + rho0
        envs, focks, cstate = self.space.build_space(qd, cfg.trunc_per_pol)
        dims, dims2, rho0 = self.space.build_qutip_space(cstate, qd.dot, focks)

        if drive is not None:
            drive = drive.with_cached_tlist(cfg.tlist)
            t_solver, Delta_eff_solver = two_photon_detuning_profile(
                qd._EL, drive, cfg.time_unit_s
            )
            if Delta_eff_solver is not None:
                t_phys = t_solver * cfg.time_unit_s
                Delta_eff_phys = Delta_eff_solver / cfg.time_unit_s

                def detuning_fn(t_phys_scalar, t=t_phys, y=Delta_eff_phys):
                    return float(np.interp(t_phys_scalar, t, y))

                drive = drive.with_detuning(detuning_fn)
        # 3) Hamiltonians + collapses
        H = self.hams.compose(qd, dims, drive, time_unit_s=cfg.time_unit_s)
        C = self.collapses.compose(qd, dims, time_unit_s=cfg.time_unit_s)

        # 4) Observables + expectation layout
        P_qd = self.observables.compose_qd(
            qd, dims, time_unit_s=cfg.time_unit_s
        )
        P_lm = self.observables.compose_lm(
            qd, dims, time_unit_s=cfg.time_unit_s
        )
        e_ops, idx = self.layout.select(P_qd, P_lm, qd)

        # 5) Solve
        result = self.solver.solve(H, rho0, cfg.tlist, C, e_ops)

        # 6) Pack QDTraces (same as in run)
        qd_tr = result.expect[idx["qd"]]
        fly_H = result.expect[idx["fly_H"]]
        fly_V = result.expect[idx["fly_V"]]
        out_H = result.expect[idx["out_H"]]
        out_V = result.expect[idx["out_V"]]
        fly_T = result.expect[idx["fly_T"]]

        Omega_t = area_t = None
        if drive is not None:
            # Tell the drive what 1.0 solver time unit equals in seconds
            coeff = drive.qutip_coeff(time_unit_s=cfg.time_unit_s)

            # Ω_solver(t') already includes the time scaling (so that ∫Ω_solver dt' = ∫Ω_phys dt_phys)
            Omega_t = np.array([coeff(t, {}) for t in cfg.tlist], dtype=float)

            area_t = np.concatenate(
                [
                    [0.0],
                    np.cumsum(
                        0.5 * np.diff(cfg.tlist) * (Omega_t[1:] + Omega_t[:-1])
                    ),
                ]
            )
            area_num = np.trapezoid(Omega_t, cfg.tlist)

        flying_labels = [
            m.label
            for m in qd.modes.modes
            if getattr(m, "source", None) == TransitionType.EXTERNAL
        ]
        intrinsic_labels = [
            m.label
            for m in qd.modes.modes
            if getattr(m, "source", None) == TransitionType.INTERNAL
        ]

        intrinsic_labels_tex = [
            m.label_tex
            for m in qd.modes.modes
            if getattr(m, "source", None) == TransitionType.INTERNAL
        ]

        traces = QDTraces(
            t=cfg.tlist,
            time_unit_s=cfg.time_unit_s,
            classical=(drive is not None),
            flying_labels=flying_labels,
            intrinsic_labels=intrinsic_labels,
            intrinsic_labels_tex=intrinsic_labels_tex,
            qd=[qd_tr[i] for i in range(4)],
            fly_H=list(fly_H),
            fly_V=list(fly_V),
            out_H=list(out_H),
            out_V=list(out_V),
            omega=Omega_t,
            area=area_t,
        )

        # 7) Extract final composite state and (optionally) the photonic reduction
        rho_final: Optional[Qobj] = getattr(result, "final_state", None)
        rho_phot_final: Optional[Qobj] = None
        if rho_final is not None and reduce_photonic:
            # dims = [QD, mode1+, mode1-, mode2+, mode2-, ...]
            keep = list(range(1, len(dims)))  # trace out subsystem 0 (QD)
            rho_phot_final = rho_final.ptrace(keep)

        return traces, rho_final, rho_phot_final
