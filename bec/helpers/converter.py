import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------

HC_eVnm = 1236.8419843320026
"""
Planck's constant times speed of light in units of eV·nm.

Useful for quick conversion between photon energy (in eV) and wavelength (in nm):
    E[eV] * λ[nm] ≈ HC_eVnm
"""

h = 6.62607015e-34
"""Planck constant [J·s]."""

ħ = h / (2.0 * np.pi)
"""Reduced Planck constant ħ = h / (2π) [J·s]."""

c = 299_792_458.0
"""Speed of light in vacuum [m/s]."""

e = 1.602176634e-19
"""Elementary charge [C]. Conversion: 1 eV = e joules."""


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def energy_to_wavelength_nm(E_eV: float) -> float:
    """
    Convert photon energy in electronvolts (eV) to wavelength in nanometers (nm).

    Parameters
    ----------
    E_eV : float
        Photon energy in electronvolts.

    Returns
    -------
    float
        Wavelength in nanometers corresponding to the given energy.
    """
    return HC_eVnm / E_eV


def wavelength_to_energy_eV(lambda_nm: float) -> float:
    """
    Convert wavelength in nanometers (nm) to photon energy in electronvolts (eV).

    Parameters
    ----------
    lambda_nm : float
        Wavelength in nanometers.

    Returns
    -------
    float
        Photon energy in electronvolts corresponding to the given wavelength.
    """
    return HC_eVnm / lambda_nm


def omega_from_lambda_nm(lambda_nm: float) -> float:
    """
    Convert wavelength in nanometers to angular frequency ω [rad/s].

    Parameters
    ----------
    lambda_nm : float
        Wavelength in nanometers.

    Returns
    -------
    float
        Angular frequency in radians per second.
    """
    lambda_m = lambda_nm * 1e-9
    freq = c / lambda_m  # Hz
    return 2.0 * np.pi * freq  # rad/s


def omega_from_energy_eV(E_eV: float) -> float:
    """
    Convert photon energy in electronvolts to angular frequency ω [rad/s].

    Parameters
    ----------
    E_eV : float
        Photon energy in electronvolts.

    Returns
    -------
    float
        Angular frequency in radians per second.
    """
    E_J = E_eV * e
    freq = E_J / h  # Hz
    return 2.0 * np.pi * freq  # rad/s
