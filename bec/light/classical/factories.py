from __future__ import annotations

from typing import Optional, Any

from bec.units import (
    QuantityLike,
    as_quantity,
    energy_to_rad_s,
    wavelength_to_energy,
    Q,
)

from bec.light.envelopes.gaussian import GaussianEnvelope
from bec.light.core.polarization import JonesMatrix, JonesState

from .amplitude import FieldAmplitude
from .carrier import Carrier
from .drive import ClassicalFieldDrive


def gaussian_field_drive(
    *,
    t0: Any,
    sigma: Optional[Any] = None,
    fwhm: Optional[Any] = None,
    E0: Any,
    # carrier specification (choose one style)
    omega0: Optional[Any] = None,  # rad/s
    wavelength: Optional[Any] = None,  # nm, um, ...
    energy: Optional[Any] = None,  # eV
    delta_omega: Any = Q(0.0, "rad/s"),
    phi0: float = 0.0,
    pol_state: Optional[JonesState] = None,
    pol_transform: Optional[JonesMatrix] = None,
    label: Optional[str] = None,
) -> ClassicalFieldDrive:
    """
    Build a ClassicalFieldDrive with a normalized Gaussian envelope and unitful field amplitude.

    Envelope:
      - GaussianEnvelope is normalized so integral f(t) dt = 1.
      - You choose t0 and either sigma or fwhm (time-like).

    Amplitude:
      - E0 is the field scale in V/m. The physical envelope is E(t) = E0 * f(t).

    Carrier:
      - optional. If omega0 is provided, uses that.
      - else if wavelength is provided, converts to energy then omega0.
      - else if energy is provided, converts to omega0.
      - delta_omega is added (can be QuantityLike in rad/s).
    """
    t0q = as_quantity(t0, "s")

    if (sigma is None) == (fwhm is None):
        raise ValueError("Provide exactly one of sigma or fwhm")

    if sigma is not None:
        env = GaussianEnvelope(t0=t0q, sigma=as_quantity(sigma, "s"))
    else:
        env = GaussianEnvelope.from_fwhm(t0=t0q, fwhm=as_quantity(fwhm, "s"))

    amp = FieldAmplitude(E0=as_quantity(E0, "V/m"))

    car = None
    if omega0 is not None or wavelength is not None or energy is not None:
        if omega0 is not None:
            w0 = as_quantity(omega0, "rad/s")
        elif wavelength is not None:
            # convert wavelength -> energy (eV) -> omega (rad/s)
            E = wavelength_to_energy(wavelength, out_unit="eV")
            w0 = energy_to_rad_s(E, out_unit="rad/s")
        else:
            w0 = energy_to_rad_s(as_quantity(energy, "eV"), out_unit="rad/s")

        car = Carrier(
            omega0=w0,
            delta_omega=as_quantity(delta_omega, "rad/s"),
            phi0=float(phi0),
        )

    return ClassicalFieldDrive(
        envelope=env,
        amplitude=amp,
        carrier=car,
        pol_state=pol_state,
        pol_transform=pol_transform,
        label=label,
    )
