# bec/quantum_dot/utils.py
import numpy as np

# Physical constants (CODATA 2018)
h = 6.62607015e-34  # Planck constant [J·s]
ħ = h / (2.0 * np.pi)  # Reduced Planck constant [J·s]
c = 299_792_458.0  # Speed of light [m/s]
e = 1.602176634e-19  # Elementary charge [C] (1 eV = e J)


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
        Photon energy in eV.

    Returns
    -------
    float
        Angular frequency in radians per second.
    """
    E_J = E_eV * e
    freq = E_J / h  # Hz
    return 2.0 * np.pi * freq  # rad/s
