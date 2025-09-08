from typing import Optional, List, Dict, Union
import numpy as np
from qutip import Qobj, basis, mesolve, tensor, qeye

from bec.light.lightmode_qutip import LightModeQuTiP
from bec.operators.qd_operators import QDState, transition_operator
from bec.operators.fock_operators import rotated_ladder_operator
from bec.quantum_dot.base import QuantumDotSystemBase
from bec.quantum_dot.params import EnergyLevels, CavityParams, DipoleParams


class QuantumDotSystem(QuantumDotSystemBase):
    """
    Concrete quantum dot system modeled using QuTiP.
    """

    def get_observables(self, **params) -> list[Qobj]:
        """
        Return list of projectors as Qobj for population monitoring.
        """
        return [
            Qobj(self.get_projectors()[QDState.G]),
            Qobj(self.get_projectors()[QDState.X1]),
            Qobj(self.get_projectors()[QDState.X2]),
            Qobj(self.get_projectors()[QDState.XX]),
        ]

    def get_projectors(self) -> Dict[QDState, Qobj]:
        """
        Return projection operators |i⟩⟨i| for each QD state as QuTiP Qobj.
        """
        return {
            state: Qobj(transition_operator(state, state)) for state in QDState
        }

    def build_hamiltonian(
        self,
        tlist: np.ndarray,
        space: List[Union[LightModeQuTiP, "QuantumDotSystem"]],
        classical_drives=None,
        wavelength_tolerance_nm=1.0,
    ):
        """
        Builds Hamiltonian according to the state space
        """
        dims = [subsystem.dim for subsystem in space]
        dim_total = np.prod(dims)
        H_total = Qobj(
            np.zeros((dim_total, dim_total), dtype=complex), dims=[dims, dims]
        )

        for i, subsystem in enumerate(space):
            H_local = subsystem.local_hamiltonian()
            H_total += self._embed_operator(H_local, i, dims)

        H_total += self._add_interactions(space, wavelength_tolerance_nm, dims)
        return H_total

    def _add_interactions(self, space, wavelength_tolerance_nm, dims):
        dim_total = np.prod(dims)
        H_int = Qobj(
            np.zeros((dim_total, dim_total), dtype=complex), dims=[dims, dims]
        )

        qd_idx = next(
            i for i, s in enumerate(space) if isinstance(s, QuantumDotSystem)
        )
        qd = space[qd_idx]

        for mode_idx, mode in enumerate(space):
            if not isinstance(mode, LightModeQuTiP):
                continue

            assert hasattr(mode, "wavelength_nm")
            assert hasattr(mode, "polarization_mode")  # e.g. 'plus', 'minus'
            assert hasattr(mode, "rotation_angle_rad")  # theta

            for from_state, to_state in [
                (QDState.XX, QDState.X1),
                (QDState.XX, QDState.X2),
                (QDState.X1, QDState.G),
                (QDState.X2, QDState.G),
            ]:
                if not self._matches_transition(
                    mode.wavelength_nm,
                    from_state,
                    to_state,
                    wavelength_tolerance_nm,
                ):
                    continue

                sigma = Qobj(transition_operator(to_state, from_state))  # QD

                # Construct rotated annihilation operator
                a_rot = rotated_ladder_operator(
                    dim=mode.fock_dim,  # assumes same dim for H and V
                    theta=mode.rotation_angle_rad,
                    mode=mode.polarization_mode,
                    operator="annihilation",
                )
                a_rot = Qobj(a_rot)

                # Embed rotated operator and sigma into full space
                op1 = self._embed_operator(
                    a_rot.dag(), mode_idx, dims
                ) @ self._embed_operator(sigma, qd_idx, dims)
                op2 = self._embed_operator(
                    a_rot, mode_idx, dims
                ) @ self._embed_operator(sigma.dag(), qd_idx, dims)

                g = getattr(mode, "coupling_strength_Hz", 1.0e6)
                H_int += 2 * np.pi * g * (op1 + op2)

        return H_int

    def _embed_operator(
        self, op: Qobj, subsystem_index: int, dims: List[int]
    ) -> Qobj:
        """Embed an operator acting on one subsystem into the full Hilbert space."""
        ops = [qeye(d) for d in dims]
        ops[subsystem_index] = op
        embedded = tensor(*ops)
        embedded.dims = [dims, dims]
        return embedded

    def local_hamiltonian(self) -> Qobj:
        """
        Returns the local Hamiltonian of the QD (4x4).
        """
        H = Qobj(np.zeros((self.dim, self.dim), dtype=complex))
        for state in QDState:
            E = self.energy_levels[state.name]
            ket = basis(4, state.value)
            H += 2 * np.pi * E * ket * ket.dag()

        if self.energy_levels["fss"] != 0:
            X1 = basis(4, QDState.X1.value)
            X2 = basis(4, QDState.X2.value)
            H += (
                np.pi
                * self.energy_levels["fss"]
                * (X1 * X1.dag() - X2 * X2.dag())
            )

        return H

    def _matches_transition(
        self,
        wavelength_nm: float,
        from_state: QDState,
        to_state: QDState,
        tolerance_nm: float = 1.0,
    ) -> bool:
        """
        Check if the given wavelength matches the energy difference between two QD states.
        """

        E1 = self.energy_levels[from_state.name]
        E2 = self.energy_levels[to_state.name]
        delta_E = abs(E1 - E2)  # in Hz

        h = 6.62607015e-34  # Planck constant
        c = 299792458  # speed of light (m/s)

        expected_lambda_m = h * c / delta_E
        expected_lambda_nm = expected_lambda_m * 1e9

        return abs(expected_lambda_nm - wavelength_nm) <= tolerance_nm
