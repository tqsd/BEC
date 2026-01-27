from __future__ import annotations

from functools import cached_property
from typing import Dict


from bec.quantum_dot.units import as_eV
from bec.units import QuantityLike, Q, c, h, hbar, magnitude


class EnergiesMixin:
    @cached_property
    def energies(self) -> Dict[str, QuantityLike]:
        el = getattr(self.qd, "energy_structure", None) or self.qd.energy_levels
        return {
            "G": Q(0.0, "eV"),
            "X1": as_eV(el.X1),
            "X2": as_eV(el.X2),
            "XX": as_eV(el.XX),
        }

    @cached_property
    def transition_energy(self) -> Dict[object, QuantityLike]:
        # local import to avoid cycles
        from bec.quantum_dot.enums import Transition

        E = self.energies
        return {
            Transition.G_X1: (E["X1"] - E["G"]).to("eV"),
            Transition.X1_G: (E["G"] - E["X1"]).to("eV"),
            Transition.G_X2: (E["X2"] - E["G"]).to("eV"),
            Transition.X2_G: (E["G"] - E["X2"]).to("eV"),
            Transition.X1_XX: (E["XX"] - E["X1"]).to("eV"),
            Transition.XX_X1: (E["X1"] - E["XX"]).to("eV"),
            Transition.X2_XX: (E["XX"] - E["X2"]).to("eV"),
            Transition.XX_X2: (E["X2"] - E["XX"]).to("eV"),
            Transition.G_XX: (E["XX"] - E["G"]).to("eV"),
            Transition.XX_G: (E["G"] - E["XX"]).to("eV"),
        }

    def _directed_transition(self, tr):
        from bec.quantum_dot.enums import TransitionPair

        if isinstance(tr, TransitionPair):
            fwd, _ = self.t_registry.directed(tr)
            return fwd
        return tr

    def omega(self, tr) -> QuantityLike:
        tr = self._directed_transition(tr)
        dE = self.transition_energy[tr].to("J")
        return (dE / hbar).to("rad/s")

    def freq(self, tr) -> QuantityLike:
        tr = self._directed_transition(tr)
        dE = self.transition_energy[tr].to("J")
        return (dE / h).to("Hz")

    def omega_abs(self, tr) -> QuantityLike:
        from bec.units import Q

        w = float(self.omega(tr).to("rad/s").magnitude)
        return abs(w) * Q(1.0, "rad/s")

    def omega_2ph_per_photon(self) -> QuantityLike:
        from bec.quantum_dot.enums import TransitionPair

        return (self.omega(TransitionPair.G_XX) / 2.0).to("rad/s")

    def wavelength_vacuum(self, tr) -> QuantityLike:
        return (c / self.freq(tr)).to("m")

    def omega_ref_rad_s(self, tr) -> float:
        """
        Return |omega(tr)| as a plain float in rad/s.
        Suitable for drive decoding / detuning calculations.

        Accepts:
          - Transition (directed) or TransitionPair (will be directed via _directed_transition)
        """
        w = self.omega(
            tr
        )  # QuantityLike rad/s (can be negative for reverse transitions)
        return float(magnitude(w, "rad/s"))
