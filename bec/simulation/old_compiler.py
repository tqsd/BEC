from typing import Optional, Sequence, Union

import numpy as np

from bec.light.classical import ClassicalCoherentDrive
from bec.operators.qd_operators import QDState
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.me.types import CollapseTermKind
from bec.simulation.protocols import (
    CollapseComposer,
    DriveDecoder,
    DriveStrengthModel,
    HamiltonianComposer,
    ObservablesComposer,
    RatesStage,
)
from bec.simulation.stages.collapse_composition import DefaultCollapseComposer
from bec.simulation.stages.drive_decoder import DefaultDriveDecoder
from bec.simulation.stages.drive_strength import DefaultDriveStrengthModel
from bec.simulation.stages.hamiltonian_composition import (
    DefaultHamiltonianComposer,
)
from bec.simulation.stages.observables_composer import (
    DefaultObservablesComposer,
)
from bec.simulation.stages.rates_stage import DefaultRatesStage
from bec.simulation.types import MEProblem, TruncationPolicy, build_dims
from bec.simulation.utils.initial_state_construction import (
    default_rho0_from_dims,
)


class MECompiler:
    def __init__(
        self,
        *,
        decoder: Optional[DriveDecoder] = None,
        drive_strength: Optional[DriveStrengthModel] = None,
        h_comp: Optional[HamiltonianComposer] = None,
        o_comp: Optional[ObservablesComposer] = None,
        c_comp: Optional[CollapseComposer] = None,
        r_stage: Optional[RatesStage] = None,
        truncation: Optional[TruncationPolicy] = None,
    ):
        self._decoder: DriveDecoder = decoder or DefaultDriveDecoder()
        self._drive_strength: DriveStrengthModel = (
            drive_strength or DefaultDriveStrengthModel()
        )
        self._h_comp: HamiltonianComposer = (
            h_comp or DefaultHamiltonianComposer()
        )
        self._r_stage: RatesStage = r_stage or DefaultRatesStage()
        self._c_comp: CollapseComposer = c_comp or DefaultCollapseComposer()
        self._o_comp: ObservablesComposer = (
            o_comp or DefaultObservablesComposer()
        )
        self._trunc: TruncationPolicy = truncation or TruncationPolicy(
            pol_dim=2
        )

    @property
    def truncation(self) -> TruncationPolicy:
        return self._trunc

    def compile(
        self,
        *,
        qd: QuantumDot,
        drives: Sequence[ClassicalCoherentDrive],
        tlist: np.ndarray,
        time_unit_s: float,
        rho0: Optional[Union[np.ndarray, QDState]] = None,
        args: Optional[dict] = None,
        meta: Optional[dict] = None,
    ) -> MEProblem:
        tlist = np.asarray(tlist, dtype=float)
        time_unit_s = float(time_unit_s)

        dims = build_dims(qd, self._trunc)
        D = int(np.prod(dims))

        if rho0 is None:
            rho0_arr = default_rho0_from_dims(dims, qd_state=QDState.G)
        elif isinstance(rho0, QDState):
            rho0_arr = default_rho0_from_dims(dims, qd_state=rho0)
        elif isinstance(rho0, np.ndarray):
            rho0_arr = np.asarray(rho0)
            if rho0_arr.shape != (D, D):
                raise ValueError(
                    f"rho0 has shape {rho0_arr.shape}, expected {(D, D)}"
                )
            if rho0_arr.dtype != np.complex128:
                rho0_arr = rho0_arr.astype(np.complex128, copy=False)
        else:
            raise Exception(
                f"Incompatible rho0 type QDState or np.ndarray expected"
            )

        resolved = self._decoder.decode(
            qd=qd, drives=drives, tlist=tlist, time_unit_s=time_unit_s
        )
        drive_coeffs = self._drive_strength.build_many(
            qd=qd, resolved=tuple(resolved), time_unit_s=time_unit_s
        )

        rates_bundle = self._r_stage.compute(
            qd=qd,
            resolved=tuple(resolved),
            drive_coeffs=drive_coeffs,
            time_unit_s=time_unit_s,
            tlist=tlist,
        )

        H_terms = self._h_comp.compose(
            qd=qd,
            resolved=resolved,
            drive_coeffs=drive_coeffs,
            dims=dims,
            time_unit_s=time_unit_s,
        )

        C_terms = self._c_comp.compose(
            qd=qd, dims=dims, rates=rates_bundle, time_unit_s=time_unit_s
        )

        obs = self._o_comp.compose(qd, dims)

        return MEProblem(
            qd=qd,
            tlist=tlist,
            time_unit_s=time_unit_s,
            dims=dims,
            rho0=rho0_arr,
            h_terms=tuple(H_terms),
            c_terms=tuple(C_terms),
            observables=obs,
            args=dict(args or {}),
            meta={
                **(meta or {}),
                "num_modes": len(qd.modes.modes),
                "num_drives": len(drives),
                "truncation": {
                    "pol_dim_default": getattr(self._trunc, "pol_dim", None)
                },
            },
        )
