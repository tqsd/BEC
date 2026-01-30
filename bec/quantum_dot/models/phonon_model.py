from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Protocol

import math
import numpy as np
from scipy.integrate import quad

from smef.core.units import (
    Q,
    QuantityLike,
    as_quantity,
    hbar as _hbar,
    kB as _kB,
)

from bec.quantum_dot.enums import QDState, RateKey, Transition
from bec.quantum_dot.transitions import (
    TransitionRegistry,
    DEFAULT_TRANSITION_REGISTRY,
)
from bec.quantum_dot.spec.phonon_params import (
    PhononModelKind,
    PhononParams,
    SpectralDensityKind,
)


@dataclass(frozen=True)
class PolaronEIDConfig:
    """
    Float-only config for downstream drive-dependent scattering calculations.

    Notes
    -----
    This config intentionally contains only floats so that time-dependent rate
    calculations can be done efficiently where you already have float arrays.

    The upstream contract remains unitful: these floats must be derived from
    unitful parameters stored on the QD.

    Parameters
    ----------
    enabled:
        If False, downstream stage should do nothing.
    alpha_s2:
        Spectral density strength in s^2.
    omega_c_rad_s:
        Cutoff frequency in rad/s.
    temperature_K:
        Temperature in Kelvin.
    """

    enabled: bool = False
    alpha_s2: float = 0.0
    omega_c_rad_s: float = 0.0
    temperature_K: float = 0.0


@dataclass(frozen=True)
class PhononOutputs:
    """
    Outputs produced by a phonon model.

    Attributes
    ----------
    rates:
        Constant rates in 1/s keyed by RateKey.
    b_polaron:
        Polaron dressing factors per transition (dimensionless floats).
    eid:
        Config for downstream drive-dependent scattering calculations.

    Notes
    -----
    - ``b_polaron`` is dimensionless and returned as float.
    - All rates are returned unitful (QuantityLike in 1/s).
    """

    rates: Dict[RateKey, QuantityLike] = field(default_factory=dict)
    b_polaron: Dict[Transition, float] = field(default_factory=dict)
    eid: PolaronEIDConfig = field(default_factory=PolaronEIDConfig)


class PhononModelProto(Protocol):
    """
    Protocol for phonon models owned by QuantumDot.
    """

    def compute(self) -> PhononOutputs: ...


class NullPhononModel:
    """
    No-op phonon model.

    Returns empty outputs.
    """

    def compute(self) -> PhononOutputs:
        return PhononOutputs()


class PolaronLAPhononModel:
    r"""
    Deformation-potential LA polaron model.

    Spectral density (super-ohmic with gaussian cutoff):

    .. math::

        J(\omega) = \alpha \omega^3 \exp(-( \omega / \omega_c )^2)

    Polaron dressing factor for a transition (i -> j) with:

    .. math::

        s^2(i,j) = (\phi_i - \phi_j)^2

    is computed as:

    .. math::

        \langle B \rangle(T) =
            \exp\left(
                -\frac{1}{2} s^2
                \int_0^\infty
                \frac{J(\omega)}{\omega^2}
                \coth\left(\frac{\beta \hbar \omega}{2}\right)
                d\omega
            \right)

    where ``beta = 1/(kB*T)``.

    Exciton relaxation (optional)
    -----------------------------
    If enabled, and if X1 and X2 couplings differ, and if an exciton splitting
    frequency ``omega_x`` is provided externally, then a minimal golden-rule
    estimate can be used:

    .. math::

        \gamma_\downarrow = 2\pi s^2 J(\omega_x) (n(\omega_x)+1)

        \gamma_\uparrow = 2\pi s^2 J(\omega_x) n(\omega_x)

    with Bose factor:

    .. math::

        n(\omega) = \frac{1}{\exp(\hbar \omega / k_B T) - 1}

    Notes
    -----
    This is a deliberately minimal physically-motivated starting point.
    You can refine it later (e.g., proper eigenbasis mixing, detailed balance
    checks, inclusion of additional scattering channels).
    """

    def __init__(
        self,
        *,
        params: PhononParams,
        transitions: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY,
        exciton_split_rad_s: Optional[QuantityLike] = None,
    ):
        self._P = params
        self._tr = transitions
        self._omega_x = exciton_split_rad_s
        self._cache: Dict[str, float] = {}

    # ---------------- state couplings ----------------

    def _phi(self, s: QDState) -> float:
        c = self._P.couplings
        if s is QDState.G:
            return float(c.phi_g)
        if s is QDState.X1:
            return float(c.phi_x1)
        if s is QDState.X2:
            return float(c.phi_x2)
        if s is QDState.XX:
            return float(c.phi_xx)
        return 0.0

    def s2_for_transition(self, tr: Transition) -> float:
        i, j = self._tr.endpoints(tr)
        d = self._phi(i) - self._phi(j)
        return float(d * d)

    # ---------------- spectral density + bose ----------------

    def _j_of_w(self, w: float) -> float:
        pol = self._P.polaron_la
        if pol.spectral_density is not SpectralDensityKind.SUPER_OHMIC_GAUSSIAN:
            return 0.0
        alpha = float(pol.alpha.to("s**2").magnitude)
        wc = float(pol.omega_c.to("rad/s").magnitude)
        if alpha <= 0.0 or wc <= 0.0 or w <= 0.0:
            return 0.0
        x = w / wc
        return float(alpha * (w**3) * math.exp(-(x * x)))

    def _bose_n(self, w: float) -> float:
        T = float(self._P.temperature.to("K").magnitude)
        if T <= 0.0 or w <= 0.0:
            return 0.0
        y_q = (_hbar * Q(w, "rad/s")) / (_kB * Q(T, "K"))
        y = float(y_q.to_base_units().magnitude)
        if y > 80.0:
            return 0.0
        ey = math.exp(y)
        return float(1.0 / (ey - 1.0))

    # ---------------- polaron B ----------------

    def polaron_b(self, *, s2: float) -> float:
        P = self._P
        if P.kind is not PhononModelKind.POLARON_LA:
            return 1.0
        if not bool(P.polaron_la.enable_polaron_renorm):
            return 1.0

        pol = P.polaron_la
        alpha = float(pol.alpha.to("s**2").magnitude)
        wc = float(pol.omega_c.to("rad/s").magnitude)
        T = float(P.temperature.to("K").magnitude)

        if alpha <= 0.0 or wc <= 0.0 or s2 <= 0.0:
            return 1.0

        key = "B:{:.6e}:{:.6e}:{:.6e}:{:.6e}".format(alpha, wc, T, float(s2))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # eta = (hbar*wc)/(2*kB*T)
        if T <= 0.0:
            eta = float("inf")
        else:
            eta_q = (_hbar * Q(wc, "rad/s")) / (2.0 * _kB * Q(T, "K"))
            eta = float(eta_q.to_base_units().magnitude)

        def coth(z: float) -> float:
            if z == 0.0:
                return float("inf")
            az = abs(z)
            if az < 1e-6:
                return (1.0 / z) + (z / 3.0)
            return 1.0 / math.tanh(z)

        def integrand(x: float) -> float:
            # dimensionless: x * exp(-x^2) * coth(eta*x)
            if x == 0.0:
                if math.isinf(eta):
                    return 0.0
                return 1.0 / eta
            if math.isinf(eta):
                return float(x * math.exp(-x * x))
            return float(x * math.exp(-x * x) * coth(eta * x))

        x_max = 8.0
        I, _ = quad(integrand, 0.0, x_max, epsabs=1e-10, epsrel=1e-8, limit=200)

        exponent = -0.5 * float(s2) * alpha * (wc * wc) * float(I)
        B = float(math.exp(exponent))

        if not math.isfinite(B):
            raise RuntimeError(
                f"polaron_b produced non-finite value: exponent={exponent}"
            )

        if B < 0.0:
            B = 0.0
        if B > 1.0:
            B = 1.0

        self._cache[key] = B
        return B

    # ---------------- constant rates ----------------

    def _phenomenological_rates(self) -> Dict[RateKey, QuantityLike]:
        ph = self._P.phenomenological
        out: Dict[RateKey, QuantityLike] = {}

        g1 = float(ph.gamma_phi_x1.to("1/s").magnitude)
        g2 = float(ph.gamma_phi_x2.to("1/s").magnitude)
        gxx = float(ph.gamma_phi_xx.to("1/s").magnitude)

        r12 = float(ph.gamma_relax_x1_x2.to("1/s").magnitude)
        r21 = float(ph.gamma_relax_x2_x1.to("1/s").magnitude)

        if g1 > 0.0:
            out[RateKey.PH_DEPH_X1] = as_quantity(g1, "1/s")
        if g2 > 0.0:
            out[RateKey.PH_DEPH_X2] = as_quantity(g2, "1/s")
        if gxx > 0.0:
            out[RateKey.PH_DEPH_XX] = as_quantity(gxx, "1/s")
        if r12 > 0.0:
            out[RateKey.PH_RELAX_X1_X2] = as_quantity(r12, "1/s")
        if r21 > 0.0:
            out[RateKey.PH_RELAX_X2_X1] = as_quantity(r21, "1/s")

        return out

    def _polaron_exciton_relax_rates(self) -> Dict[RateKey, QuantityLike]:
        P = self._P
        if not bool(P.polaron_la.enable_exciton_relaxation):
            return {}

        if self._omega_x is None:
            return {}
        w = float(as_quantity(self._omega_x, "rad/s").magnitude)
        if w <= 0.0:
            return {}

        # requires X1 and X2 to be distinguishable
        d = self._phi(QDState.X1) - self._phi(QDState.X2)
        s2 = float(d * d)
        if s2 <= 0.0:
            return {}

        Jw = self._j_of_w(w)
        if Jw <= 0.0:
            return {}

        n = self._bose_n(w)
        g_down = float(2.0 * math.pi * s2 * Jw * (n + 1.0))
        g_up = float(2.0 * math.pi * s2 * Jw * n)

        out: Dict[RateKey, QuantityLike] = {}
        if g_down > 0.0:
            out[RateKey.PH_RELAX_X1_X2] = as_quantity(g_down, "1/s")
        if g_up > 0.0:
            out[RateKey.PH_RELAX_X2_X1] = as_quantity(g_up, "1/s")
        return out

    # ---------------- public compute ----------------

    def compute(self) -> PhononOutputs:
        P = self._P
        if P.kind is PhononModelKind.NONE:
            return PhononOutputs()

        rates: Dict[RateKey, QuantityLike] = {}
        rates.update(self._phenomenological_rates())

        bmap: Dict[Transition, float] = {}
        if (
            P.kind is PhononModelKind.POLARON_LA
            and P.polaron_la.enable_polaron_renorm
        ):
            # compute B only for transitions you care about; caller can extend later
            for tr in self._tr.transitions():
                s2 = self.s2_for_transition(tr)
                bmap[tr] = self.polaron_b(s2=s2)

        if P.kind is PhononModelKind.POLARON_LA:
            rates.update(self._polaron_exciton_relax_rates())

        pol = P.polaron_la
        eid = PolaronEIDConfig(
            enabled=bool(
                P.kind is PhononModelKind.POLARON_LA and pol.enable_eid
            ),
            alpha_s2=float(pol.alpha.to("s**2").magnitude),
            omega_c_rad_s=float(pol.omega_c.to("rad/s").magnitude),
            temperature_K=float(P.temperature.to("K").magnitude),
        )

        return PhononOutputs(rates=rates, b_polaron=bmap, eid=eid)
