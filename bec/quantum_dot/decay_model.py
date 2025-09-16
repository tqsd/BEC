from typing import Dict
import numpy as np
from scipy.constants import epsilon_0, hbar, c, e


class DecayModel:
    def __init__(
        self,
        energy_levels_dict: Dict[str, float],
        cavity_params,
        dipole_params,
    ):
        self.el = energy_levels_dict
        self.cavity = cavity_params
        self.dipole = dipole_params
        if not self.dipole:
            raise ValueError("DipoleParams required to compute gammas.")

    def _omega(self, Ei_eV: float, Ef_eV: float) -> float:
        dE_eV = float(Ei_eV - Ef_eV)
        if dE_eV <= 0:
            return 0.0
        return (dE_eV * e) / hbar

    def _purcell(self, omega: float) -> float:
        if not self.cavity or omega <= 0.0:
            return 0.0
        lam = 2 * np.pi * c / omega
        n = float(self.cavity.n)
        Vm = float(self.cavity.Veff_um3) * 1e-18
        Q = float(self.cavity.Q)
        return max((3.0 / (4.0 * np.pi**2)) * (lam / n) ** 3 * (Q / Vm), 0.0)

    def _gamma0(self, omega: float, mu_Cm: float) -> float:
        if omega <= 0.0 or mu_Cm <= 0.0:
            return 0.0
        return (omega**3 * mu_Cm**2) / (3.0 * np.pi * epsilon_0 * hbar * c**3)

    def compute(self) -> Dict[str, float]:
        mu = float(self.dipole.dipole_moment_Cm)

        def gamma(Ei, Ef):
            w = self._omega(Ei, Ef)
            g = self._gamma0(w, mu) * (1.0 + self._purcell(w))  # 1/s
            return g

        el = self.el
        return {
            "L_XX_X1": gamma(el["XX"], el["X1"]),
            "L_XX_X2": gamma(el["XX"], el["X2"]),
            "L_X1_G": gamma(el["X1"], el["G"]),
            "L_X2_G": gamma(el["X2"], el["G"]),
        }
