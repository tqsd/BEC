from __future__ import annotations

from typing import Any, Optional

from smef.core.units import (
    Q,
    QuantityLike,
    as_quantity,
    energy_to_rad_s,
    magnitude,
    wavelength_to_energy,
)

from bec.light.core.polarization import JonesMatrix, JonesState
from bec.light.envelopes.gaussian import GaussianEnvelopeU

from .amplitude import FieldAmplitude
from .carrier import Carrier
from .field_drive import ClassicalFieldDriveU


def gaussian_field_drive(
    *,
    t0: Any,
    sigma: Optional[Any] = None,
    fwhm: Optional[Any] = None,
    E0: Any,
    # carrier specification (choose one style)
    omega0: Optional[Any] = None,  # rad/s
    wavelength: Optional[Any] = None,  # nm, um, m, ...
    energy: Optional[Any] = None,  # eV
    delta_omega: Any = Q(0.0, "rad/s"),
    phi0: float = 0.0,
    pol_state: Optional[JonesState] = None,
    pol_transform: Optional[JonesMatrix] = None,
    preferred_kind: Optional[str] = None,  # "1ph" or "2ph"
    label: Optional[str] = None,
) -> ClassicalFieldDriveU:
    """
    Build a ClassicalFieldDriveU with a peak-normalized Gaussian envelope and unitful field amplitude.

    Envelope:
      - GaussianEnvelopeU is peak-normalized: max g(t) = 1.
      - Provide t0 and exactly one of sigma or fwhm (time-like).

    Amplitude:
      - E0 is the peak field scale in V/m.
      - Physical envelope magnitude is E_env(t) = E0 * g(t).

    Carrier:
      - optional. If omega0 is provided, uses that.
      - else if wavelength is provided, converts to energy (eV) then omega0 (rad/s).
      - else if energy is provided, converts to omega0 (rad/s).
      - delta_omega is added (rad/s).
    """
    t0q = as_quantity(t0, "s")

    if (sigma is None) == (fwhm is None):
        raise ValueError("Provide exactly one of sigma or fwhm")

    if sigma is not None:
        env = GaussianEnvelopeU(t0=t0q, sigma=as_quantity(sigma, "s"))
    else:
        env = GaussianEnvelopeU.from_fwhm(t0=t0q, fwhm=as_quantity(fwhm, "s"))

    amp = FieldAmplitude(E0=as_quantity(E0, "V/m"))

    car = None
    if omega0 is not None or wavelength is not None or energy is not None:
        if omega0 is not None:
            w0 = as_quantity(omega0, "rad/s")
        elif wavelength is not None:
            # wavelength -> energy (eV) -> omega (rad/s)
            E = wavelength_to_energy(wavelength, out_unit="eV")
            w0 = energy_to_rad_s(E, out_unit="rad/s")
        else:
            w0 = energy_to_rad_s(as_quantity(energy, "eV"), out_unit="rad/s")

        car = Carrier(
            omega0=w0,
            delta_omega=as_quantity(delta_omega, "rad/s"),
            phi0=float(phi0),
        )

    return ClassicalFieldDriveU(
        envelope=env,
        amplitude=amp,
        carrier=car,
        pol_state=pol_state,
        pol_transform=pol_transform,
        preferred_kind=preferred_kind,
        label=label,
    )
