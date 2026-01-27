from dataclasses import dataclass
from typing import Any, Optional
from bec.units import (
    QuantityLike,
    as_quantity,
    energy_to_wavelength,
    wavelength_to_energy,
    energy_to_rad_s,
)


@dataclass(frozen=True)
class LightChannel:
    key: Any
    label: Optional[str] = None
    label_tex: Optional[str] = None

    energy: Optional[QuantityLike] = None  # eV
    wavelength: Optional[QuantityLike] = None  # nm
    omega: Optional[QuantityLike] = None  # rad/s

    def resolved_energy(self) -> Optional[QuantityLike]:
        if self.energy is not None:
            return as_quantity(self.energy, "eV")
        if self.wavelength is not None:
            return wavelength_to_energy(self.wavelength, out_unit="eV")
        if self.omega is not None:
            # E = ħ ω
            # you can add a helper in bec.units for rad/s -> eV; for now:
            from bec.units import hbar

            E_J = (hbar * as_quantity(self.omega, "rad/s")).to("J")
            return E_J.to("eV")
        return None

    def resolved_wavelength(self) -> Optional[QuantityLike]:
        if self.wavelength is not None:
            return as_quantity(self.wavelength, "nm")
        E = self.resolved_energy()
        if E is None:
            return None
        return energy_to_wavelength(E, out_unit="nm")

    def resolved_omega(self) -> Optional[QuantityLike]:
        if self.omega is not None:
            return as_quantity(self.omega, "rad/s")
        E = self.resolved_energy()
