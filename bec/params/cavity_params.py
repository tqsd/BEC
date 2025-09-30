from dataclasses import dataclass


@dataclass
class CavityParams:
    r"""
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

    .. math::
        F_p = \frac{3}{4\pi^2}\Big(\frac{\lamdba}{n}\Big)^3 \frac{Q}{V_{eff}}

    where lambda is the wavelength in meters, n is the refractive index, Q is the
    quality factor, and V_eff is the effective mode volume in m^3.
    """

    Q: float
    Veff_um3: float
    lambda_nm: float
    n: float = 3.5
