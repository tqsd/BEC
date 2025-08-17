from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Any, List
import numpy as np
from scipy.constants import epsilon_0, hbar, c, e

from bec.operators.qd_operators import QDState, transition_operator
from .params import EnergyLevels, CavityParams, DipoleParams


class QuantumDotSystemBase(ABC):
    """
    Abstract base class for a four-level quantum dot system.
    State space: |G>, |X1>, |X2>, |XX>.
    """

    def __init__(
        self,
        energy_levels: EnergyLevels,
        initial_state: QDState = QDState.G,
        cavity_params: Optional[CavityParams] = None,
        dipole_params: Optional[DipoleParams] = None,
        time_unit_s: float = 1e-9,
    ):
        self.initial_state = initial_state
        self._energy_levels = energy_levels
        self.energy_levels = energy_levels.compute()  # dict of energies
        self.cavity_params = cavity_params
        self.dipole_params = dipole_params
        self.dim = len(QDState)
        self.time_unit_s = time_unit_s

        # Calculate internal angle θ from fine structure splitting
        self._modes = energy_levels.modes()
        self.THETA, self.PHI, self._exitation_eigs = (
            energy_levels.exciton_rotation_params()
        )
        self.gammas = self._compute_gammas()

    def _omega_from_eV(self, Ei_eV: float, Ef_eV: float) -> float:
        """Angular frequency (rad/s) for transition Ei->Ef, energies in eV."""
        dE_eV = float(Ei_eV - Ef_eV)
        if dE_eV <= 0:
            return 0.0
        return (dE_eV * e) / hbar

    def _purcell_factor(self, omega_rad_s: float) -> float:
        """Purcell factor for this transition; 0 if no cavity params."""
        if not self.cavity_params or omega_rad_s <= 0.0:
            return 0.0
        lam = 2 * np.pi * c / omega_rad_s  # wavelength in m
        n = float(self.cavity_params.n)
        Vm_m3 = float(self.cavity_params.Veff_um3) * 1e-18  # μm^3 -> m^3
        Q = float(self.cavity_params.Q)
        # canonical Purcell form; using *transition* wavelength
        Fp = (3.0 / (4.0 * np.pi**2)) * (lam / n) ** 3 * (Q / Vm_m3)
        return max(Fp, 0.0)

    def _gamma_free_space(self, omega_rad_s: float, mu_Cm: float) -> float:
        """Free-space spontaneous emission rate in 1/s."""
        if omega_rad_s <= 0.0 or mu_Cm <= 0.0:
            return 0.0
        return (omega_rad_s**3 * mu_Cm**2) / (
            3.0 * np.pi * epsilon_0 * hbar * c**3
        )

    # --- main computation -----------------------------------------------------

    def _compute_gammas(self) -> dict[str, float]:
        """
        Return {'L_XX_X1', 'L_XX_X2', 'L_X1_G', 'L_X2_G'} as *dimensionless*
        rates in your simulation time units (i.e., s^-1 * time_unit_s).
        """
        if not self.dipole_params:
            raise ValueError(
                "DipoleParams(dipole_moment_Cm=...) required to compute gammas."
            )

        el = self.energy_levels  # dict: 'G','X1','X2','XX',...
        mu = float(self.dipole_params.dipole_moment_Cm)

        def gamma_sim(Ei, Ef) -> float:
            omega = self._omega_from_eV(Ei, Ef)
            gamma0 = self._gamma_free_space(omega, mu)  # 1/s
            # unitless
            Fp = self._purcell_factor(omega)
            gamma_si = gamma0 * (1.0 + Fp)  # 1/s
            # convert to 1/(sim unit)
            return gamma_si * self.time_unit_s

        gammas = {
            "L_XX_X1": gamma_sim(el["XX"], el["X1"]),
            "L_XX_X2": gamma_sim(el["XX"], el["X2"]),
            "L_X1_G": gamma_sim(el["X1"], el["G"]),
            "L_X2_G": gamma_sim(el["X2"], el["G"]),
        }
        return gammas

    def get_projectors(self) -> Dict[QDState, np.ndarray]:
        """
        Return projection operators |i⟩⟨i| for each QD state as NumPy arrays.
        """
        return {state: transition_operator(state, state) for state in QDState}

    def get_transition_op(
        self, from_state: QDState, to_state: QDState
    ) -> np.ndarray:
        """
        Return transition operator |to⟩⟨from| as a NumPy array.
        """
        return transition_operator(from_state, to_state)

    def get_collapse_operators(self, **params) -> list[np.ndarray]:
        """
        Return Lindblad collapse operators.
        """
        pass

    def get_observables(self, **params) -> list[np.ndarray]:
        """
        Return list of observables (e.g. projectors).
        """
        pass

    def validate_light_mode_for_transition(
        self,
        light_mode: object,
        from_state: QDState,
        to_state: QDState,
        tolerance_nm: float = 1.0,
    ) -> bool:
        """
        Validate whether the light mode wavelength matches the transition energy
        (from_state → to_state) within a specified tolerance in nm.

        Parameters
        ----------
        light_mode : object
            The LightMode instance. Must have .wavelength_nm attribute.
        from_state : QDState
            Initial QD state in the transition.
        to_state : QDState
            Final QD state in the transition.
        tolerance_nm : float
            Allowed deviation in nanometers.

        Returns
        -------
        bool
            True if the light mode matches the transition energy, False otherwise.
        """
        # Transition energy (in Hz)
        E_i = self.energy_levels[from_state.name]
        E_f = self.energy_levels[to_state.name]
        delta_E = abs(E_i - E_f)  # in Hz

        # Expected wavelength (in m)
        h = 6.62607015e-34  # Planck constant
        c = 299792458  # speed of light
        expected_lambda_m = h * c / delta_E
        expected_lambda_nm = expected_lambda_m * 1e9

        return (
            abs(expected_lambda_nm - light_mode.wavelength_nm) <= tolerance_nm
        )
