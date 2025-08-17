from dataclasses import dataclass
from typing import List, Optional


import numpy as np

from bec.helpers.converter import energy_to_wavelength_nm
from bec.light.base import LightMode


@dataclass
class EnergyLevels:
    exciton: float  # average exciton energy (e.g., in eV)
    biexciton: float  # absolute biexciton energy (eV)
    fss: float  # fine-structure splitting Δ = E_X1 − E_X2 (eV)
    delta_prime: float = 0.0  # cross FSS term Δ' (eV)

    def compute(self) -> dict:
        """Return energy levels of the four QD states as a dict."""
        X1 = self.exciton + self.fss / 2
        X2 = self.exciton - self.fss / 2
        binding_energy = (X1 + X2) - self.biexciton
        return {
            "G": 0.0,
            "X1": X1,
            "X2": X2,
            "XX": self.biexciton,
            "fss": self.fss,
            "delta_prime": self.delta_prime,
            "binding_energy": binding_energy,
        }

    def modes(self, wavelength_tolerance_nm=1e-6) -> List[LightMode]:
        levels = self.compute()
        modes = []

        # Helper to compare wavelengths within tolerance
        def same_lambda(e1, e2):
            return (
                abs(energy_to_wavelength_nm(e1) - energy_to_wavelength_nm(e2))
                < wavelength_tolerance_nm
            )

        # Energies of the four relevant transitions
        e_X1_G = levels["X1"] - levels["G"]
        e_X2_G = levels["X2"] - levels["G"]
        e_XX_X1 = levels["XX"] - levels["X1"]
        e_XX_X2 = levels["XX"] - levels["X2"]

        def etw(e):
            return energy_to_wavelength_nm(e)

        if abs(self.fss) < 1e-9 and abs(self.delta_prime) < 1e-9:
            modes.append(LightMode(etw(e_X1_G), "internal", [0, 1], "X<->G"))
            modes.append(LightMode(etw(e_XX_X1), "internal", [2, 3], "XX<->X"))
        elif same_lambda(e_XX_X1, e_X2_G):
            print("SAME LAMBDA")
            # Cross-degenerate case: XX→X1 same λ as X2→G
            modes.append(
                LightMode(etw(e_X1_G), "internal", [0, 3], "X1<->G & X2<->XX")
            )
            modes.append(
                LightMode(etw(e_X2_G), "internal", [1, 2], "X2<->G & X1<->XX")
            )
        else:
            # General case: all four distinct
            modes.append(LightMode(etw(e_X1_G), "internal", [0], "X1<->G"))
            modes.append(LightMode(etw(e_X2_G), "internal", [1], "X2<->G"))
            modes.append(LightMode(etw(e_XX_X1), "internal", [2], "XX<->X1"))
            modes.append(LightMode(etw(e_XX_X2), "internal", [3], "XX<->X2"))

        return modes

    def exciton_rotation_params(self):
        delta = self.fss
        delta_p = self.delta_prime
        # Hamiltonian in |X1>, |X2> basis
        H = 0.5 * \
            np.array([[delta, delta_p], [delta_p, -delta]], dtype=complex)

        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(H)

        # First eigenvector defines rotation (up to global phase)
        v = eigvecs[:, 0]
        theta = np.arctan2(abs(v[1]), abs(v[0]))
        phi = np.angle(v[1]) - np.angle(v[0])

        return theta, phi, eigvals


@dataclass
class CavityParams:
    Q: float
    Veff_um3: float
    lambda_nm: float
    n: float = 3.5


@dataclass
class DipoleParams:
    dipole_moment_Cm: float  # in C·m
