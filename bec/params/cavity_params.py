from dataclasses import dataclass


@dataclass
class CavityParams:
    """
    Optical cavity parameters for quantum-dot simulations.

    Parameters
    ----------
    Q : float
        Quality factor of the cavity (dimensionless). Determines linewidth
        and photon lifetime.
    Veff_um3 : float
        Effective mode volume of the cavity in cubic micrometers (µm³).
    lambda_nm : float
        Resonant cavity wavelength in nanometers (nm).
    n : float, default=3.5
        Refractive index of the cavity material. Defaults to 3.5,
        representative of GaAs and related III–V semiconductors.

    Notes
    -----
    These parameters are typically used in Purcell factor calculations:

        F_p = (3 / (4π²)) * (λ / n)³ * (Q / V_eff)

    where λ is the wavelength in meters, n is the refractive index, Q is the
    quality factor, and V_eff is the effective mode volume in m³.
    """

    Q: float
    Veff_um3: float
    lambda_nm: float
    n: float = 3.5
