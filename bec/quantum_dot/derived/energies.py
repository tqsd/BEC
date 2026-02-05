from __future__ import annotations

from functools import cached_property

from smef.core.units import Q, QuantityLike, c, h, hbar, magnitude

from bec.core.units import as_eV
from bec.quantum_dot.enums import QDState


class EnergiesMixin:
    @cached_property
    def energies(self) -> dict[str, QuantityLike]:
        el = self.qd.energy
        return {
            QDState.G: Q(0.0, "eV"),
            QDState.X1: as_eV(el.X1),
            QDState.X2: as_eV(el.X2),
            QDState.XX: as_eV(el.XX),
        }

    @cached_property
    def transition_energy(self) -> dict[object, QuantityLike]:
        # Local import to avoid cycles
        from bec.quantum_dot.enums import Transition

        E = self.energies
        return {
            Transition.G_X1: (E[QDState.X1] - E[QDState.G]).to("eV"),
            Transition.X1_G: (E[QDState.G] - E[QDState.X1]).to("eV"),
            Transition.G_X2: (E[QDState.X2] - E[QDState.G]).to("eV"),
            Transition.X2_G: (E[QDState.G] - E[QDState.X2]).to("eV"),
            Transition.X1_XX: (E[QDState.XX] - E[QDState.X1]).to("eV"),
            Transition.XX_X1: (E[QDState.X1] - E[QDState.XX]).to("eV"),
            Transition.X2_XX: (E[QDState.XX] - E[QDState.X2]).to("eV"),
            Transition.XX_X2: (E[QDState.X2] - E[QDState.XX]).to("eV"),
            Transition.G_XX: (E[QDState.XX] - E[QDState.G]).to("eV"),
            Transition.XX_G: (E[QDState.G] - E[QDState.XX]).to("eV"),
        }

    def _directed_transition(self, tr):
        from bec.quantum_dot.enums import TransitionPair

        if isinstance(tr, TransitionPair):
            fwd, _ = self.t_registry.directed(tr)
            return fwd
        return tr

    def omega(self, tr) -> QuantityLike:
        tr = self._directed_transition(tr)
        dE_J = self.transition_energy[tr].to("J")
        return (dE_J / hbar).to("rad/s")

    def freq(self, tr) -> QuantityLike:
        tr = self._directed_transition(tr)
        dE_J = self.transition_energy[tr].to("J")
        return (dE_J / h).to("Hz")

    def omega_abs(self, tr) -> QuantityLike:
        w = float(self.omega(tr).to("rad/s").magnitude)
        return Q(abs(w), "rad/s")

    def omega_2ph_per_photon(self) -> QuantityLike:
        from bec.quantum_dot.enums import TransitionPair

        return (self.omega(TransitionPair.G_XX) / 2.0).to("rad/s")

    def wavelength_vacuum(self, tr) -> QuantityLike:
        return (c / self.freq(tr)).to("m")

    def omega_ref_rad_s_energy(self, tr) -> float:
        """
        Return |omega(tr)| as a plain float in rad/s.

        Accepts:
          - Transition (directed) or TransitionPair (will be directed to forward)
        """
        w = self.omega(tr)
        return abs(float(magnitude(w, "rad/s")))

    def energy(self, st: QDState) -> QuantityLike:
        """
        Return absolute energy of a dot state as QuantityLike.

        Must be convertible to eV or J.
        """
        ep = self.qd.energy_levels  # or whatever your params object is called

        if st is QDState.G:
            return as_quantity(getattr(ep, "E_G", 0.0), "eV")
        if st is QDState.X1:
            return as_quantity(ep.E_X1, "eV")
        if st is QDState.X2:
            return as_quantity(ep.E_X2, "eV")
        if st is QDState.XX:
            return as_quantity(ep.E_XX, "eV")
        raise KeyError(st)
