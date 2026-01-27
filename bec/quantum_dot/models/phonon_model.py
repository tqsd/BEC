from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.integrate import quad

from bec.quantum_dot.parameters.phonons import PhononModelType, PhononParams
from bec.units import (
    Q,
    QuantityLike,
    as_quantity,
    magnitude,
    hbar as _hbar,
    kB as _kB,
)
from bec.quantum_dot.enums import QDState, Transition, TransitionPair
from bec.quantum_dot.transitions import (
    DEFAULT_REGISTRY,
    RateKey,
    TransitionRegistry,
)


@dataclass(frozen=True)
class PhononOutputs:
    B_polaron_per_transition: Dict[Transition, float] = field(
        default_factory=dict
    )
    rates: Dict[RateKey, QuantityLike] = field(default_factory=dict)


class PhononModel:
    """
    Unitful phonon model.

    - Returns B (polaron dressing) as dimensionless float.
    - Returns rates as pint quantities in 1/s.
    - Does NOT pass pint objects into scipy.integrate.quad (quad expects floats).
    """

    def __init__(
        self,
        *,
        phonon_params: Optional[PhononParams] = None,
        transitions: TransitionRegistry = DEFAULT_REGISTRY,
    ):
        self._P = phonon_params
        self._tr = transitions
        self._cache: Dict[str, float] = {}

    # ---------------- polaron dressing ----------------

    def polaron_B(self, *, s2: float = 1.0) -> float:
        """
        Compute thermal polaron dressing factor <B>(T).

        Uses J(w) = alpha * w^3 * exp(-(w/wc)^2) where:
          - alpha has units s^2
          - wc has units rad/s
        Here we compute the standard dimensionless integral form and keep quad float-only.
        """
        P = self._P
        if P is None:
            return 1.0
        if getattr(P, "model", None) != PhononModelType.POLARON:
            return 1.0
        if not bool(getattr(P, "enable_polaron_renorm", False)):
            return 1.0

        alpha = float(getattr(P, "alpha_s2", getattr(P, "alpha", 0.0)))
        wc = magnitude(getattr(P, "omega_c_rad_s"), "rad/s")
        T = magnitude(getattr(P, "temperature_K"), "K")

        if alpha <= 0.0 or wc <= 0.0:
            return 1.0

        key = "B:{:.6e}:{:.6e}:{:.6e}:{:.6e}".format(alpha, wc, T, s2)
        if key in self._cache:
            return self._cache[key]

        # eta = (hbar*wc)/(2*kB*T) dimensionless
        if T <= 0.0:
            eta = np.inf
        else:
            eta_q = (_hbar * Q(wc, "rad/s")) / (2.0 * _kB * Q(T, "K"))
            eta = float(eta_q.to_base_units().magnitude)

        def coth(z: float) -> float:
            if z == 0.0:
                return np.inf
            az = abs(z)
            if az < 1e-6:
                return (1.0 / z) + (z / 3.0)
            return 1.0 / np.tanh(z)

        def integrand(x: float) -> float:
            # dimensionless integrand: x * exp(-x^2) * coth(eta*x)
            if x == 0.0:
                if np.isinf(eta):
                    return 0.0
                return 1.0 / eta
            if np.isinf(eta):
                return float(x * np.exp(-x * x))
            return float(x * np.exp(-x * x) * coth(eta * x))

        x_max = 8.0
        I, _ = quad(integrand, 0.0, x_max, epsabs=1e-10, epsrel=1e-8, limit=200)

        exponent = -0.5 * (s2 * alpha * (wc * wc) * I)
        B = float(np.exp(exponent))

        if not np.isfinite(B):
            raise RuntimeError(
                "polaron_B produced non-finite value (exponent={})".format(
                    exponent
                )
            )

        if B < 0.0:
            B = 0.0
        if B > 1.0:
            B = 1.0

        self._cache[key] = B
        return B

    # ---------------- state-dependent coupling ----------------
    def _phi_for_state(self, s: QDState) -> float:
        """
        Return the dimensionless deformation-potential coupling parameter for a QD level.
        This assumes your PhononParams stores phi_G, phi_X, phi_XX as dimensionless floats.
        """
        P = self._P
        if P is None:
            return 0.0
        if s == QDState.G:
            return float(getattr(P, "phi_G", 0.0))
        if s in (QDState.X1, QDState.X2):
            return float(getattr(P, "phi_X", 0.0))
        if s == QDState.XX:
            return float(getattr(P, "phi_XX", 0.0))
        return 0.0

    def _s2_for_transition(self, tr: Transition) -> float:
        """
        s^2 = (phi_i - phi_j)^2 using endpoints from TransitionRegistry.
        """
        i, j = self._tr.endpoints(tr)
        d = self._phi_for_state(i) - self._phi_for_state(j)
        return float(d * d)

    def polaron_B_for_transition(self, tr: Transition) -> float:
        P = self._P
        if P is None:
            return 1.0
        if getattr(P, "model", None) != PhononModelType.POLARON:
            return 1.0
        if not bool(getattr(P, "enable_polaron_renorm", False)):
            return 1.0
        return self.polaron_B(s2=self._s2_for_transition(tr))

    # ---------------- public compute ----------------

    def compute(self) -> PhononOutputs:
        P = self._P
        if P is None:
            return PhononOutputs()

        rates: Dict[RateKey, QuantityLike] = {}

        # If these are already in 1/s, enforce units:
        g_xp = getattr(P, "gamma_phi_Xp_1_s", 0.0)
        g_xm = getattr(P, "gamma_phi_Xm_1_s", 0.0)
        g_xx = getattr(P, "gamma_phi_XX_1_s", 0.0)

        if float(g_xp) > 0.0:
            rates[RateKey.PH_DEPH_X1] = as_quantity(g_xp, "1/s")
        if float(g_xm) > 0.0:
            rates[RateKey.PH_DEPH_X2] = as_quantity(g_xm, "1/s")
        if float(g_xx) > 0.0:
            # If you want XX dephasing as a separate key, add it to RateKey.
            # For now, you can store it under one of your existing keys or create RateKey.PH_DEPH_XX.
            pass

        Bmap: Dict[Transition, float] = {}
        if getattr(P, "model", None) == PhononModelType.POLARON and bool(
            getattr(P, "enable_polaron_renorm", False)
        ):
            # Only compute for physically relevant transitions if you prefer:
            for tp in (
                TransitionPair.G_X1,
                TransitionPair.G_X2,
                TransitionPair.X1_XX,
                TransitionPair.X2_XX,
                TransitionPair.G_XX,
            ):
                fwd, bwd = self._tr.directed(tp)
                Bmap[fwd] = self.polaron_B_for_transition(fwd)
                Bmap[bwd] = self.polaron_B_for_transition(bwd)

        return PhononOutputs(B_polaron_per_transition=Bmap, rates=rates)
