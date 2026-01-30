from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Mapping, Optional, Sequence

import numpy as np

from smef.core.units import QuantityLike, magnitude

from bec.quantum_dot.enums import RateKey, Transition, TransitionPair
from bec.quantum_dot.models.decay_model import DecayOutputs
from bec.quantum_dot.models.phonon_model import PhononOutputs
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.exciton_mixing_params import ExcitonMixingParams
from bec.quantum_dot.spec.phonon_params import PhononParams
from bec.quantum_dot.transitions import TransitionRegistry


@dataclass(frozen=True)
class QDDerivedView:
    energy: EnergyStructure
    dipoles: DipoleParams
    mixing: Optional[ExcitonMixingParams]
    phonons: Optional[PhononParams]

    t_registry: TransitionRegistry
    decay_outputs: DecayOutputs
    phonon_outputs: PhononOutputs
    rates: Mapping[RateKey, QuantityLike]

    # ---------- drive pipeline contract: strength ----------

    def mu(self, tr: Transition) -> QuantityLike:
        return self.dipoles.mu(tr)

    def mu_Cm(self, tr: Transition) -> float:
        return float(magnitude(self.dipoles.mu(tr), "C*m"))

    def polaron_B(self, tr: Transition) -> float:
        """
        Polaron renormalization factor B(tr) for this specific directed transition.
        Returns 1.0 if unavailable/disabled.
        """
        po = self.phonon_outputs
        if po is None:
            return 1.0

        bmap = getattr(po, "b_polaron", None)
        if not isinstance(bmap, Mapping):
            return 1.0

        try:
            v = bmap.get(tr, 1.0)
            b = float(v)
        except Exception:
            return 1.0

        if not np.isfinite(b):
            return 1.0

        # clamp into [0, 1] (typical physical range)
        if b < 0.0:
            return 0.0
        if b > 1.0:
            return 1.0
        return b

    # ---------- drive pipeline contract: decoder ----------

    def drive_targets(self) -> Sequence[TransitionPair]:
        out = []
        for pair in self.t_registry.pairs():
            if self.t_registry.spec(pair).drive_allowed:
                out.append(pair)
        return tuple(out)

    def polaron_B_pair(self, pair: TransitionPair) -> float:
        fwd, _ = self.t_registry.directed(pair)
        return self.polaron_B(fwd)

    def drive_kind(self, pair: TransitionPair) -> str:
        spec = self.t_registry.spec(pair)
        return "2ph" if int(spec.order) == 2 else "1ph"

    def omega_ref_rad_s(self, pair: TransitionPair) -> float:
        from smef.core.units import Q, hbar

        fwd, _ = self.t_registry.directed(pair)
        src, dst = self.t_registry.endpoints(fwd)

        e_src = float(magnitude(getattr(self.energy, src.name), "eV"))
        e_dst = float(magnitude(getattr(self.energy, dst.name), "eV"))
        de_eV = float(e_dst - e_src)

        de_J = Q(de_eV, "eV").to("J")
        return float((de_J / hbar).to("rad/s").magnitude)

    def drive_projection(self, tr: Transition, E: np.ndarray) -> complex:
        E = np.asarray(E, dtype=complex).reshape(2)
        mu_hat = np.asarray(self.dipoles.e_pol_hv(tr), dtype=complex).reshape(2)
        return complex(np.vdot(mu_hat, E))

    @cached_property
    def fss_eV(self) -> float:
        e_x1 = float(magnitude(self.energy.X1, "eV"))
        e_x2 = float(magnitude(self.energy.X2, "eV"))
        return float(e_x1 - e_x2)

    @cached_property
    def gamma_phi_eid_scale(self) -> float:
        ph = self.phonons
        if ph is None:
            return 0.0
        val = getattr(ph.phenomenological, "gamma_phi_eid_scale", 0.0)
        try:
            return float(val)
        except Exception:
            return 0.0
