from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad
from smef.core.units import (
    Q,
    QuantityLike,
    as_quantity,
)
from smef.core.units import (
    hbar as _hbar,
)
from smef.core.units import (
    kB as _kB,
)

from bec.quantum_dot.enums import QDState, RateKey, Transition, TransitionPair
from bec.quantum_dot.spec.exciton_mixing_params import ExcitonMixingParams
from bec.quantum_dot.spec.phonon_params import PhononModelType, PhononParams
from bec.quantum_dot.transitions import (
    DEFAULT_TRANSITION_REGISTRY,
    TransitionRegistry,
)


@dataclass(frozen=True)
class PolaronEIDConfig:
    """
    Minimal float-only configuration for drive-dependent dephasing calculations.

    This is intentionally "dumb data" so the drive emitter can compute:
        J(w) = alpha * w^3 * exp(-(w/wc)^2)
        coth(beta*hbar*w/2)
    using float arrays.

    s2 should be computed per transition from (phi_i - phi_j)^2.
    """

    enabled: bool = False
    alpha_s2: float = 0.0
    omega_c_rad_s: float = 0.0
    temperature_K: float = 0.0


@dataclass(frozen=True)
class PhononOutputs:
    B_polaron_per_transition: dict[Transition, float] = field(
        default_factory=dict
    )
    rates: dict[RateKey, QuantityLike] = field(default_factory=dict)
    polaron_eid: PolaronEIDConfig = PolaronEIDConfig()


class PhononModel:
    """
    Unitful phonon model with float-only integration.

    - Returns B (polaron dressing) as dimensionless float.
    - Returns constant rates as quantities in 1/s (phenomenological only).
    - Provides polaron EID config (floats) for time-dependent rate calculation
      elsewhere (e.g., in QDDriveTermEmitter).
    - Keeps scipy.integrate.quad float-only.
    """

    def __init__(
        self,
        *,
        phonon_params: PhononParams | None = None,
        exciton_splitting: ExcitonMixingParams | None = None,
        transitions: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY,
    ):
        self._EM = exciton_splitting
        self._P = phonon_params
        self._tr = transitions
        self._cache: dict[str, float] = {}

    # ---------------- polaron dressing ----------------

    def polaron_B(self, *, s2: float = 1.0) -> float:
        """
        Compute thermal polaron dressing factor <B>(T).

        Spectral density: J(w) = alpha * w^3 * exp(-(w/wc)^2)
        with alpha [s^2], wc [rad/s].

        Returns float in [0, 1].
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

        key = f"B:{alpha:.6e}:{wc:.6e}:{T:.6e}:{float(s2):.6e}"
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
                f"polaron_B produced non-finite value (exponent={exponent})"
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

        if s is QDState.X1:
            if getattr(P, "phi_X1", None) is not None:
                return float(P.phi_X1)
            if getattr(P, "phi_X", None) is not None:
                return float(P.phi_X)
            return 0.0

        if s is QDState.X2:
            if getattr(P, "phi_X2", None) is not None:
                return float(P.phi_X2)
            if getattr(P, "phi_X", None) is not None:
                return float(P.phi_X)
            return 0.0

        if s in (QDState.X1, QDState.X2):
            return float(P.phi_X)
        if s is QDState.XX:
            return float(P.phi_XX)
        return 0.0

    def s2_for_transition(self, tr: Transition) -> float:
        """
        s2 = (phi_i - phi_j)^2 using endpoints from TransitionRegistry.
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
        return self.polaron_B(s2=self.s2_for_transition(tr))

    # ---------------- helpers for phenomenological rates ----------------

    def _phenomenological_rates(self) -> dict[RateKey, QuantityLike]:
        P = self._P
        if P is None:
            return {}

        out: dict[RateKey, QuantityLike] = {}

        g_xp = float(P.phenomenological.gamma_phi_Xp_1_s)
        g_xm = float(P.phenomenological.gamma_phi_Xm_1_s)
        g_xx = float(P.phenomenological.gamma_phi_XX_1_s)
        g_12 = float(P.phenomenological.gamma_relax_X1_X2_1_s)
        g_21 = float(P.phenomenological.gamma_relax_X2_X1_1_s)

        if g_xp > 0.0:
            out[RateKey.PH_DEPH_X1] = as_quantity(g_xp, "1/s")
        if g_xm > 0.0:
            out[RateKey.PH_DEPH_X2] = as_quantity(g_xm, "1/s")
        if g_xx > 0.0:
            out[RateKey.PH_DEPH_XX] = as_quantity(g_xx, "1/s")
        if g_12 > 0.0:
            out[RateKey.PH_RELAX_X1_X2] = as_quantity(g_12, "1/s")
        if g_21 > 0.0:
            out[RateKey.PH_RELAX_X2_X1] = as_quantity(g_21, "1/s")

        return out

    # ---------------- public compute ----------------

    def compute(self) -> PhononOutputs:
        P = self._P
        if P is None:
            return PhononOutputs()

        # Start with phenomenological rates always available if nonzero.
        rates = self._phenomenological_rates()

        # Handle explicit phenom-only model
        if P.model is PhononModelType.PHENOMENOLOGICAL:
            return PhononOutputs(
                B_polaron_per_transition={},
                rates=rates,
                polaron_eid=PolaronEIDConfig(enabled=False),
            )

        # Unknown model: safest no-op (but keep phenom rates if present)
        if P.model is not PhononModelType.POLARON:
            return PhononOutputs(
                B_polaron_per_transition={},
                rates=rates,
                polaron_eid=PolaronEIDConfig(enabled=False),
            )

        # ---- POLARON path ----
        Bmap: dict[Transition, float] = {}
        if P.polaron.enable_polaron_renorm:
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

        alpha = float(P.polaron.alpha_s2)
        wc = float(P.polaron.omega_c_rad_s)
        T = float(P.temperature_K)

        eid_enabled = bool(P.polaron.enable_eid)
        pol_eid = PolaronEIDConfig(
            enabled=bool(eid_enabled and alpha > 0.0 and wc > 0.0 and T >= 0.0),
            alpha_s2=float(max(alpha, 0.0)),
            omega_c_rad_s=float(max(wc, 0.0)),
            temperature_K=float(max(T, 0.0)),
        )

        pol_eid = PolaronEIDConfig(
            enabled=bool(eid_enabled and alpha > 0.0 and wc > 0.0 and T >= 0.0),
            alpha_s2=float(max(alpha, 0.0)),
            omega_c_rad_s=float(max(wc, 0.0)),
            temperature_K=float(max(T, 0.0)),
        )

        return PhononOutputs(
            B_polaron_per_transition=Bmap,
            rates=rates,  # keep (or set {} if you want exclusive)
            polaron_eid=pol_eid,
        )
