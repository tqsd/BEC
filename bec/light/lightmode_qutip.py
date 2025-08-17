from typing import Callable
from qutip import Qobj, tensor, identity
import numpy as np
from bec.operators.fock_operators import rotated_ladder_operator


class LightModeQuTiP:
    def __init__(
        self,
        wavelength_nm: float,
        fock_dim: int,
        rotation_angle_rad: float,
        polarization_mode: str = "plus",
        coupling_strength_Hz: float = 1e6,
    ):
        self.wavelength_nm = wavelength_nm
        self.fock_dim = fock_dim
        self.rotation_angle_rad = rotation_angle_rad
        self.polarization_mode = polarization_mode  # 'plus' or 'minus'
        self.coupling_strength_Hz = coupling_strength_Hz

        self.dim = fock_dim**2  # H ⊗ V

    def local_hamiltonian(self) -> Qobj:
        """
        Light modes are considered free fields with zero-energy offset here.
        You could later add photon energy terms.
        """
        return Qobj(np.zeros((self.dim, self.dim), dtype=complex))

    def get_annihilation_operator(self) -> Callable[[float], Qobj]:
        return lambda theta: Qobj(
            rotated_ladder_operator(
                dim=self.fock_dim, theta=theta, operator="annihilation"
            )
        )

    def get_creation_operator(self) -> Callable[[float], Qobj]:
        return lambda theta: Qobj(
            rotated_ladder_operator(
                dim=self.fock_dim, theta=theta, operator="creation"
            )
        )
