from dataclasses import dataclass

import numpy as np
from smef.core.units import (
    Q,
    QuantityLike,
    as_quantity,
    magnitude,
)
from smef.core.units import (
    c as _c,
)
from smef.core.units import (
    epsilon_0 as _eps_0,
)
from smef.core.units import (
    hbar as _hbar,
)

from bec.core.units import as_eV
from bec.quantum_dot.enums import QDState, RateKey, Transition
from bec.quantum_dot.spec.cavity_params import CavityParams
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.transitions import (
    DEFAULT_TRANSITION_REGISTRY,
    RAD_RATE_TO_TRANSITION,
    TransitionRegistry,
)


@dataclass(frozen=True)
class DecayModel:
    energy_structure: EnergyStructure
    dipole_params: DipoleParams
    cavity_params: CavityParams | None = None
    transitions: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY

    def level_energy(self, s: QDState) -> QuantityLike:
        return as_eV(getattr(self.energy_structure, s.name)).to("eV")

    def dE_photon(self, tr: Transition) -> QuantityLike:
        i, f = self.transitions.endpoints(tr)
        dE = (self.level_energy(i) - self.level_energy(f)).to("eV")
        if magnitude(dE, "eV") <= 0.0:
            return Q(0.0, "eV")
        return dE

    def omega(self, tr: Transition) -> QuantityLike:
        dE = self.dE_photon(tr)
        if magnitude(dE, "eV") <= 0.0:
            return Q(0.0, "rad/s")
        return (dE.to("J") / _hbar).to("rad/s")

    def gamma0(self, tr: Transition) -> QuantityLike:
        omega = as_quantity(self.omega(tr), "rad/s")
        mu = as_quantity(self.dipole_params.mu(tr), "C*m")
        if magnitude(omega, "rad/s") <= 0.0 or magnitude(mu, "C*m") <= 0.0:
            return Q(0.0, "1/s")
        num = omega**3 * mu**2
        den = 3.0 * np.pi * _eps_0 * _hbar * _c**3
        return (num / den).to("1/s")

    def purcell_factor(self, tr: Transition) -> float:
        cp = self.cavity_params
        if cp is None:
            return 0.0

        omega = self.omega(tr)
        if magnitude(omega, "rad/s") <= 0.0:
            return 0.0

        Qfac = float(getattr(cp, "Q", 0.0))
        n = float(getattr(cp, "n", 1.0))
        if Qfac <= 0.0 or n <= 0.0:
            return 0.0

        lam = (2.0 * np.pi * _c / omega).to("m")

        if hasattr(cp, "Veff_m3"):
            Vm = as_quantity(getattr(cp, "Veff_m3"), "m^3")
        else:
            Veff_um3 = float(getattr(cp, "Veff_um3", 0.0))
            if Veff_um3 <= 0.0:
                return 0.0
            Vm = Q(Veff_um3, "um^3").to("m^3")

        Fp = (3.0 / (4.0 * np.pi**2)) * (lam / n) ** 3 * (Qfac / Vm)
        return max(float(Fp.to("").magnitude), 0.0)

    def gamma(self, tr: Transition) -> QuantityLike:
        # Respect registry: only allow families with decay_allowed=True
        spec = self.transitions.spec(tr)
        if not spec.decay_allowed:
            return Q(0.0, "1/s")

        g0 = self.gamma0(tr)
        Fp = self.purcell_factor(tr)
        return (g0 * (1.0 + Fp)).to("1/s")

    def compute_q(self) -> dict[RateKey, QuantityLike]:
        out: dict[RateKey, QuantityLike] = {}
        for key, tr in RAD_RATE_TO_TRANSITION.items():
            out[key] = self.gamma(tr)
        return out

    def compute(self) -> dict[RateKey, float]:
        out_q = self.compute_q()
        return {k: float(v.to("1/s").magnitude) for k, v in out_q.items()}
