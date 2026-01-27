from __future__ import annotations

from bec.units import magnitude


class FastMixin:
    def omega_rad_s(self, tr) -> float:
        return magnitude(self.omega(tr), "rad/s")

    def freq_Hz(self, tr) -> float:
        return magnitude(self.freq(tr), "Hz")

    def wavelength_m(self, tr) -> float:
        return magnitude(self.wavelength_vacuum(tr), "m")
