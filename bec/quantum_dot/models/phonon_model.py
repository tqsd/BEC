from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.integrate import quad

from bec.quantum_dot.enums import QDState, RateKey, Transition, TransitionPair
from bec.quantum_dot.spec.phonon_params import PhononModelType, PhononParams
from bec.quantum_dot.transitions import (
    DEFAULT_TRANSITION_REGISTRY,
    TransitionRegistry,
)
from smef.core.units import (
    Q,
    QuantityLike,
    as_quantity,
    hbar as _hbar,
    kB as _kB,
)


@dataclass(frozen=True)
class PhononOutputs:
    B_polaron_per_transition: Dict[Transition, float] = field(
        default_factory=dict
    )
    rates: Dict[RateKey, QuantityLike] = field(default_factory=dict)


class PhononModel:
    """
    Unitful phonon model with float-only integration.

    - Returns B (polaron dressing) as dimensionless float.
    - Returns rates as quantities in 1/s.
    - Keeps scipy.integrate.quad float-only.
    """

    def __init__(
        self,
        *,
        phonon_params: Optional[PhononParams] = None,
        transitions: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY,
    ):
        self._P = phonon_params
        self._tr = transitions
        self._cache: Dict[str, float] = {}

    # ---------------- polaron dressing ----------------

    def polaron_B(self, *, s2: float = 1.0) -> float:
        """
        Compute thermal polaron dressing factor <B>(T).

        Spectral density: J(w) = alpha * w^3 * exp(-(w/wc)^2)
        with alpha [s^2], wc [rad/s].

        This computes a standard dimensionless integral and returns a float.
        """
        P = self._P
        if P is None:
            return 1.0
        if P.model is not PhononModelType.POLARON:
            return 1.0
        if not P.polaron.enable_polaron_renorm:
            return 1.0

        alpha = float(P.polaron.alpha_s2)
        wc = float(P.polaron.omega_c_rad_s)
        T = float(P.temperature_K)

        if alpha <= 0.0 or wc <= 0.0:
            return 1.0

        key = "B:{:.6e}:{:.6e}:{:.6e}:{:.6e}".format(alpha, wc, T, float(s2))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

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
        I, _ = quad(
            integrand,
            0.0,
            x_max,
            epsabs=1e-10,
            epsrel=1e-8,
            limit=200,
        )

        exponent = -0.5 * (float(s2) * alpha * (wc * wc) * float(I))
        B = float(np.exp(exponent))

        if not np.isfinite(B):
            raise RuntimeError(
                "polaron_B produced non-finite value (exponent={})".format(
                    exponent
                )
            )

        # Clamp to [0, 1] for numerical safety
        if B < 0.0:
            B = 0.0
        if B > 1.0:
            B = 1.0

        self._cache[key] = B
        return B

    # ---------------- state-dependent coupling ----------------

    def _phi_for_state(self, s: QDState) -> float:
        """
        Dimensionless deformation-potential coupling parameter for a QD level.

        PhononParams stores phi_G, phi_X, phi_XX as floats (dimensionless).
        """
        P = self._P
        if P is None:
            return 0.0
        if s is QDState.G:
            return float(P.phi_G)
        if s in (QDState.X1, QDState.X2):
            return float(P.phi_X)
        if s is QDState.XX:
            return float(P.phi_XX)
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
        if P.model is not PhononModelType.POLARON:
            return 1.0
        if not P.polaron.enable_polaron_renorm:
            return 1.0
        return self.polaron_B(s2=self._s2_for_transition(tr))

    # ---------------- public compute ----------------

    def compute(self) -> PhononOutputs:
        P = self._P
        if P is None:
            return PhononOutputs()

        rates: Dict[RateKey, QuantityLike] = {}

        # Phenomenological pure dephasing terms (already validated, unitful)
        g_xp = float(P.phenomenological.gamma_phi_Xp_1_s)
        g_xm = float(P.phenomenological.gamma_phi_Xm_1_s)
        g_xx = float(P.phenomenological.gamma_phi_XX_1_s)
        g_12 = float(P.phenomenological.gamma_relax_X1_X2_1_s)
        g_21 = float(P.phenomenological.gamma_relax_X2_X1_1_s)

        if g_xp > 0.0:
            rates[RateKey.PH_DEPH_X1] = as_quantity(g_xp, "1/s")
        if g_xm > 0.0:
            rates[RateKey.PH_DEPH_X2] = as_quantity(g_xm, "1/s")
        if g_xx > 0.0:
            rates[RateKey.PH_DEPH_XX] = as_quantity(g_xx, "1/s")
        if g_12 > 0.0:
            rates[RateKey.PH_RELAX_X1_X2] = as_quantity(g_12, "1/s")
        if g_21 > 0.0:
            rates[RateKey.PH_RELAX_X2_X1] = as_quantity(g_21, "1/s")

        Bmap: Dict[Transition, float] = {}
        if (
            P.model is PhononModelType.POLARON
            and P.polaron.enable_polaron_renorm
        ):
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
