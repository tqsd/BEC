from typing import Dict
import numpy as np
from scipy.constants import epsilon_0, hbar, c, e


class DecayModel:
    r"""
    Compute radiative decay rates (gammas) for a quantum dot with
    optional cavity enhancement (Purcell effect).

    The model uses:
    - Transition angular frequency from level energies in eV
      .. math::
        \omega = \frac{(E_i - E_f)e}{\hbar}

    - Free-space spontaneous emission rate:
      .. math::
        \Gamma_0(\omega, \mu) = \frac{\omega^3 \mu^2}{3\pi\epsilon_0\hbar c^3}

    - Purcell factor for a cavity:
      .. math::
        F_P = \frac{3}{4\pi^2}(\frac{\lambda}{n})^3\frac{Q}{V_m}

    - Total rate:
      .. math::
        \gamma = \Gamma_0(\omega, \mu) (1+F_P)

    Notes
    -----
    - Units:
      * Energies: eV
      * Dipole mont mu: C*m
      * Refractive index n: unitless
      * Veff_um3: cubic micrometers (um^3)
      * Output rates: 1/s
    - The module is not ment to be used by itself, it is embedded
    into `QuantumDot`.

    """

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
        """
        Computes transition angular frequency from initial and final
        energies.

        Parameters
        ----------
        Ei_eV: float
            Initial level energy in eV
        Ef_ev: float
            Final level energy in eV

        Returns
        -------
        float
            Angular frequency omega in rad/s. Returns 0.0 if Ei_eV <= Ef_eV.
        """
        dE_eV = float(Ei_eV - Ef_eV)
        if dE_eV <= 0:
            return 0.0
        return (dE_eV * e) / hbar

    def _purcell(self, omega: float) -> float:
        """
        Compute Purcell enhancement factor for a given angular.

        Parameters
        ----------
        omega: float
           Angular frequency in rad/s

        Returns
        -------
        float
             Dimensionlesspurcell factor F_P. 0 if no cavity params provided
        """
        if not self.cavity or omega <= 0.0:
            return 0.0
        lam = 2 * np.pi * c / omega
        n = float(self.cavity.n)
        Vm = float(self.cavity.Veff_um3) * 1e-18
        Q = float(self.cavity.Q)
        return max((3.0 / (4.0 * np.pi**2)) * (lam / n) ** 3 * (Q / Vm), 0.0)

    def _gamma0(self, omega: float, mu_Cm: float) -> float:
        """
        Free-space spontaneous emission rate

        Parameters
        ----------
        omega: float
            Angular frequency in rad/s
        mu_Cm: float
            Dipole momentum in C*m

        Returns
        -------
        float
            Free-space spontaeous emission rate in 1/s
        """
        if omega <= 0.0 or mu_Cm <= 0.0:
            return 0.0
        return (omega**3 * mu_Cm**2) / (3.0 * np.pi * epsilon_0 * hbar * c**3)

    def compute(self) -> Dict[str, float]:
        """
        Computes decay rates for XX->X1, XX->X2, X1->G, X2->G.

        Returns
        -------
        Dict[str, float]
            Mapping of decay labels to rates in 1/s:
            - "L_XX_X1": XX->X1
            - "L_XX_X2": XX->X2
            - "L_X1_G": X1->G
            - "L_X2_G": X2->G

        Notes:
          - `omega` is derived from `energy_levels_dict`,
          - `mu` is taken fro `dipole_params.dipole_moment_Cm`,
          - `F_P` is computed from `cavity_params` if present.
        """
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
