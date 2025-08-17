import numpy as np
from scipy.constants import epsilon_0, c, hbar, pi


def vacuum_gamma(lambda_nm: float, dipole_m_C: float) -> float:
    """
    Calculate the spontaneous emission rate in vacuum (γ₀) for a given transition.

    Parameters
    ----------
    lambda_nm : float
        Transition wavelength in nanometers (nm).
    dipole_m_C : float
        Transition dipole moment in Coulomb·meters (C·m).

    Returns
    -------
    float
        Spontaneous emission rate in s⁻¹ (Hz).
    """
    lambda_m = lambda_nm * 1e-9
    omega = 2 * pi * c / lambda_m
    gamma = (omega**3 * dipole_m_C**2) / (3 * pi * epsilon_0 * hbar * c**3)
    return gamma


def purcell_factor(
    lambda_em_nm: float, lambda_cav_nm: float, Q: float, Veff_um3: float, n: float = 3.5
) -> float:
    """
    Calculate the Purcell factor for a given emitter-cavity system.

    Parameters
    ----------
    lambda_em_nm : float
        Emission wavelength of the quantum emitter in nanometers (nm).
    lambda_cav_nm : float
        Resonant wavelength of the cavity in nanometers (nm).
    Q : float
        Quality factor of the cavity.
    Veff_um3 : float
        Effective mode volume of the cavity in micrometers cubed (μm³).
    n : float, optional
        Refractive index of the material (default is 3.5 for GaAs).

    Returns
    -------
    float
        Purcell enhancement factor (dimensionless).
    """
    lambda_em = lambda_em_nm * 1e-9
    lambda_cav = lambda_cav_nm * 1e-9
    Veff_m3 = Veff_um3 * 1e-18
    delta = (lambda_em - lambda_cav) / lambda_cav
    denom = 1 + 4 * Q**2 * delta**2
    F = (3 / (4 * np.pi**2)) * (lambda_em / n) ** 3 * (Q / Veff_m3) * (1 / denom)
    return F
