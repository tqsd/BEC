from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

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

from bec.quantum_dot.enums import QDState, Transition
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.exciton_mixing_params import ExcitonMixingParams
from bec.quantum_dot.spec.phonon_params import (
    PhononModelKind,
    PhononParams,
    SpectralDensityKind,
)
from bec.quantum_dot.transitions import (
    DEFAULT_TRANSITION_REGISTRY,
    RateKey,
    TransitionRegistry,
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

    rates: dict[RateKey, QuantityLike] = field(default_factory=dict)
    b_polaron: dict[Transition, float] = field(default_factory=dict)
    eid: PolaronEIDConfig = field(default_factory=PolaronEIDConfig)
    polaron_rates: PolaronDriveRates | None = None


@dataclass(frozen=True)
class _BathFftCache:
    # tau grid (uniform)
    tau_s: np.ndarray
    dtau_s: float
    # omega grid for rfft (rad/s), non-negative
    omega_grid_rad_s: np.ndarray
    # dtau * FFT(f) approximates integral of f(t) exp(-i omega t) dt
    dtau_rfft_G: np.ndarray  # complex
    dtau_rfft_one_minus: np.ndarray  # complex (will be real-ish if input real)


def _interp_complex(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray
) -> np.ndarray:
    # linear interpolation for complex arrays using real+imag parts
    xr = np.interp(x, xp, fp.real)
    xi = np.interp(x, xp, fp.imag)
    return xr + 1j * xi


def _build_fft_cache(
    tau_s: np.ndarray,
    G_tau: np.ndarray,
    one_minus: np.ndarray,
) -> _BathFftCache:
    tau = np.asarray(tau_s, dtype=float).reshape(-1)
    if tau.size < 2:
        raise ValueError("tau grid too small")

    dtau = float(tau[1] - tau[0])
    if dtau <= 0.0:
        raise ValueError("tau step must be > 0")

    diffs = np.diff(tau)
    if not np.allclose(diffs, dtau, rtol=1e-6, atol=0.0):
        raise ValueError("tau grid must be uniform for FFT method")

    N = int(tau.size)

    G = np.asarray(G_tau, dtype=complex).reshape(-1)
    om = np.asarray(one_minus, dtype=float).reshape(-1)

    if G.size != N or om.size != N:
        raise ValueError("cache arrays must match tau size")

    # rfft frequencies in cycles/s -> rad/s
    freq_hz = np.fft.rfftfreq(N, d=dtau)
    omega_grid = 2.0 * np.pi * freq_hz

    # rfft does not support complex input: do real+imag separately and recombine.
    G_re = np.ascontiguousarray(G.real, dtype=float)
    G_im = np.ascontiguousarray(G.imag, dtype=float)
    om_re = np.ascontiguousarray(om, dtype=float)

    F_re = np.fft.rfft(G_re)
    F_im = np.fft.rfft(G_im)
    dtau_rfft_G = dtau * (F_re + 1j * F_im)

    dtau_rfft_one_minus = dtau * np.fft.rfft(om_re)

    return _BathFftCache(
        tau_s=tau,
        dtau_s=dtau,
        omega_grid_rad_s=omega_grid,
        dtau_rfft_G=dtau_rfft_G,
        dtau_rfft_one_minus=dtau_rfft_one_minus,
    )


def _re_int_exp_iwt_from_fft(
    w_rad_s: np.ndarray,
    *,
    fft_cache: _BathFftCache,
) -> np.ndarray:
    """
    Approximate Re int_0^T exp(+i w t) G(t) dt.

    We have dtau*rfft(G) ~ int_0^T G(t) exp(-i w t) dt
    So for exp(+i w t): use conjugate:
      int G(t) exp(+i w t) dt ~ conj( int G(t) exp(-i w t) dt )
    """
    w = np.asarray(w_rad_s, dtype=float).reshape(-1)
    wg = fft_cache.omega_grid_rad_s
    F = fft_cache.dtau_rfft_G

    w_clip = np.clip(w, float(wg[0]), float(wg[-1]))
    val_minus = _interp_complex(w_clip, wg, F)
    val_plus = np.conjugate(val_minus)
    return np.real(val_plus).astype(float)


def _int_cos_from_fft(
    w_rad_s: np.ndarray,
    *,
    fft_cache: _BathFftCache,
) -> np.ndarray:
    """
    Approximate int_0^T one_minus(t) cos(w t) dt
    = Re int one_minus(t) exp(+i w t) dt
    """
    w = np.asarray(w_rad_s, dtype=float).reshape(-1)
    wg = fft_cache.omega_grid_rad_s
    F = fft_cache.dtau_rfft_one_minus

    w_clip = np.clip(w, float(wg[0]), float(wg[-1]))
    val_minus = _interp_complex(w_clip, wg, F)
    val_plus = np.conjugate(val_minus)
    return np.real(val_plus).astype(float)


@dataclass(frozen=True)
class _BathCache:
    tau_s: np.ndarray  # shape (Nt,)
    G_tau: np.ndarray  # shape (Nt,) complex
    # shape (Nt,) real (for cross-deph if desired)
    one_minus_exp_minus_phi: np.ndarray


_GLOBAL_BATH_CACHE: Dict[
    Tuple[float, float, float, int, float, float, int], "_BathCache"
] = {}


@dataclass(frozen=True)
class _BathCache:
    tau_s: np.ndarray  # shape (Nt,)
    G_tau: np.ndarray  # shape (Nt,) complex
    one_minus_exp_minus_phi: np.ndarray  # shape (Nt,) float
    fft: Optional["_BathFftCache"] = None


@dataclass(frozen=True)
class PolaronDriveRates:
    """
    Float-only helpers for drive-dependent phonon rates.

    Rates are returned in 1/s (physical units as floats).
    """

    enabled: bool
    alpha_s2: float
    omega_c_rad_s: float
    temperature_K: float

    _bath_cache: object = None

    def _ensure_bath_cache(
        self,
        *,
        Nt: int = 4096,
        x_max: float = 8.0,
        tau_max_factor: float = 8.0,
    ) -> _BathCache:
        # First: instance cache
        if self._bath_cache is not None:
            return self._bath_cache  # type: ignore[return-value]

        alpha = float(self.alpha_s2)
        wc = float(self.omega_c_rad_s)
        T = float(self.temperature_K)

        # Disabled / degenerate => trivial cache
        if (not self.enabled) or alpha <= 0.0 or wc <= 0.0:
            tau_s = np.linspace(0.0, 1.0, 8, dtype=float)
            G_tau = np.zeros_like(tau_s, dtype=complex)
            om = np.zeros_like(tau_s, dtype=float)
            cache = _BathCache(
                tau_s=tau_s,
                G_tau=G_tau,
                one_minus_exp_minus_phi=om,
                fft=_build_fft_cache(tau_s, G_tau, om),
            )
            object.__setattr__(self, "_bath_cache", cache)
            return cache

        # Second: global cache (critical for sweeps that rebuild QD objects)
        # Note: round floats to make keys stable against tiny print/convert noise.
        key = (
            float(f"{alpha:.16e}"),
            float(f"{wc:.16e}"),
            float(f"{T:.16e}"),
            int(Nt),
            float(f"{x_max:.16e}"),
            float(f"{tau_max_factor:.16e}"),
            4096,  # Nx (kept as constant below)
        )
        cached = _GLOBAL_BATH_CACHE.get(key)
        if cached is not None:
            object.__setattr__(self, "_bath_cache", cached)
            return cached

        # ---------------- Build bath ----------------

        Nx = 4096
        x = np.linspace(0.0, float(x_max), Nx, dtype=float)
        omega = wc * x  # rad/s

        # J(omega)/omega^2 = alpha * omega * exp(-x^2)
        jw_over_w2 = alpha * omega * np.exp(-(x * x))

        # coth(hbar*omega/(2 kB T))
        if T <= 0.0:
            coth = np.ones_like(omega, dtype=float)
        else:
            xarg_q = (_hbar * Q(omega, "rad/s")) / (2.0 * _kB * Q(T, "K"))
            xarg = np.asarray(xarg_q.to_base_units().magnitude, dtype=float)
            coth = self._coth(xarg)

        # tau grid: correlations decay on scale ~ 1/wc
        tau_max_s = float(tau_max_factor) / wc
        tau_s = np.linspace(0.0, tau_max_s, int(Nt), dtype=float)

        # Build phi(tau) for all tau using broadcasting
        # This is still the heavy step, but with global caching it happens
        # once per (alpha, wc, T, Nt, x_max, tau_max_factor).
        wt = omega[None, :] * tau_s[:, None]
        cos_wt = np.cos(wt)
        sin_wt = np.sin(wt)

        # Integral over omega grid expressed in x, with d omega = wc d x:
        re_phi = (
            np.trapezoid(
                jw_over_w2[None, :] * coth[None, :] * cos_wt, x, axis=1
            )
            * wc
        )
        im_phi = -np.trapezoid(jw_over_w2[None, :] * sin_wt, x, axis=1) * wc
        phi = re_phi + 1j * im_phi

        G_tau = np.exp(phi) - 1.0

        # Real-only variant used for cosine integral
        one_minus_exp_minus_phi = 1.0 - np.exp(-re_phi)

        fft = _build_fft_cache(tau_s, G_tau, one_minus_exp_minus_phi)

        cache = _BathCache(
            tau_s=tau_s,
            G_tau=np.asarray(G_tau, dtype=complex),
            one_minus_exp_minus_phi=np.asarray(
                one_minus_exp_minus_phi, dtype=float
            ),
            fft=fft,
        )

        _GLOBAL_BATH_CACHE[key] = cache
        object.__setattr__(self, "_bath_cache", cache)
        return cache

    def gamma_dressed_rates_1_s(
        self,
        *,
        omega_solver: np.ndarray,
        detuning_rad_s: np.ndarray,
        time_unit_s: float,
        b_polaron: float = 1.0,
        Nt: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (gamma_down, gamma_up, gamma_cd) in 1/s as float arrays.

        Fast path:
        - Uses an FFT-based lookup of int exp(+- i w tau) G(tau) d tau
        - Uses an FFT-based lookup of int cos(w tau) one_minus(tau) d tau
        """
        if not bool(self.enabled):
            n = int(np.asarray(detuning_rad_s).size)
            z = np.zeros((n,), dtype=float)
            return z, z, z

        s = float(time_unit_s)
        if s <= 0.0:
            raise ValueError("time_unit_s must be > 0")

        omega_solver = np.asarray(omega_solver, dtype=complex).reshape(-1)
        detuning_rad_s = np.asarray(detuning_rad_s, dtype=float).reshape(-1)
        if omega_solver.size != detuning_rad_s.size:
            raise ValueError(
                "detuning_rad_s must have same length as omega_solver"
            )

        cache = self._ensure_bath_cache(Nt=Nt)
        if cache.fft is None:
            # Should not happen because we always build FFT now,
            # but keep a safe fallback.
            tau_s = cache.tau_s
            G_tau = cache.G_tau
            one_minus = cache.one_minus_exp_minus_phi

            wstar = self.omega_star_rad_s(
                omega_solver,
                detuning_rad_s,
                time_unit_s=s,
                b_polaron=b_polaron,
            )
            OmR = self._omega_rabi_rad_s(
                omega_solver,
                time_unit_s=s,
                b_polaron=b_polaron,
            )
            pref = 0.5 * (OmR * OmR)

            re_pos = self._re_int_exp_iwt_G(wstar, tau_s=tau_s, G_tau=G_tau)
            re_neg = self._re_int_exp_iwt_G(-wstar, tau_s=tau_s, G_tau=G_tau)

            g_down = pref * re_pos
            g_up = pref * re_neg

            phase = tau_s[:, None] * wstar[None, :]
            cos_wt = np.cos(phase)
            val_cd = np.trapezoid(one_minus[:, None] * cos_wt, tau_s, axis=0)
            g_cd = pref * np.real(val_cd).astype(float)

            return (
                np.maximum(g_down, 0.0),
                np.maximum(g_up, 0.0),
                np.maximum(g_cd, 0.0),
            )

        fft = cache.fft

        wstar = self.omega_star_rad_s(
            omega_solver,
            detuning_rad_s,
            time_unit_s=s,
            b_polaron=b_polaron,
        )

        OmR = self._omega_rabi_rad_s(
            omega_solver,
            time_unit_s=s,
            b_polaron=b_polaron,
        )
        pref = 0.5 * (OmR * OmR)

        # Compute both Re int exp(+i w t) G(t) dt and Re int exp(-i w t) G(t) dt
        # with a single interpolation (val_minus) and conjugation.
        w = np.asarray(wstar, dtype=float).reshape(-1)
        wg = fft.omega_grid_rad_s
        F = fft.dtau_rfft_G

        w_clip = np.clip(w, float(wg[0]), float(wg[-1]))
        # int G(t) exp(-i w t) dt
        val_minus = _interp_complex(w_clip, wg, F)
        # int G(t) exp(+i w t) dt
        val_plus = np.conjugate(val_minus)

        re_pos = np.real(val_plus).astype(float)
        re_neg = np.real(val_minus).astype(float)

        g_down = pref * re_pos
        g_up = pref * re_neg

        # Cross-deph from FFT of one_minus:
        cd_int = _int_cos_from_fft(w_clip, fft_cache=fft)
        g_cd = pref * cd_int

        g_down = np.maximum(g_down, 0.0)
        g_up = np.maximum(g_up, 0.0)
        g_cd = np.maximum(g_cd, 0.0)
        return g_down, g_up, g_cd

    def _j_of_w(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        out = np.zeros_like(w)
        alpha = float(self.alpha_s2)
        wc = float(self.omega_c_rad_s)
        if alpha <= 0.0 or wc <= 0.0:
            return out
        m = w > 0.0
        x = w[m] / wc
        out[m] = alpha * (w[m] ** 3) * np.exp(-(x * x))
        return out

    @staticmethod
    def _coth(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable coth(x) for x >= 0.

        For small x, coth(x) ~ 1/x + x/3. We clamp to avoid divergence.
        """
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)

        # thresholds chosen for stability; adjust if you like
        small = x < 1e-6
        large = x > 50.0  # coth(x) -> 1 quickly

        # small-x series
        xs = x[small]
        out[small] = (1.0 / np.maximum(xs, 1e-12)) + (xs / 3.0)

        # large-x approx
        out[large] = 1.0

        # regular region
        mid = (~small) & (~large)
        xm = x[mid]
        out[mid] = 1.0 / np.tanh(xm)

        return out

    def gamma_eid_1_s(
        self,
        omega_solver: np.ndarray,
        omega_star_rad_s: np.ndarray,
        *,
        time_unit_s: float,
        calibration: float = 1.0,
        w_floor_rad_s: float = 1.0e9,
        prefactor: float = 1.0,
    ) -> np.ndarray:
        wstar = np.asarray(omega_star_rad_s, dtype=float).reshape(-1)

        if not bool(self.enabled):
            return np.zeros_like(wstar)

        s = float(time_unit_s)
        if s <= 0.0:
            raise ValueError("time_unit_s must be > 0")

        omega_solver = np.asarray(omega_solver, dtype=complex).reshape(-1)
        if omega_solver.size != wstar.size:
            raise ValueError(
                "omega_star_rad_s must have same length as omega_solver"
            )

        # Omega in physical rad/s
        omega_rad_s = omega_solver / s
        om2 = (omega_rad_s.real * omega_rad_s.real) + (
            omega_rad_s.imag * omega_rad_s.imag
        )

        # floor to avoid division by zero
        w_eff = np.maximum(np.abs(wstar), float(w_floor_rad_s))

        # J(w) and J(w)/w^2
        jw = self._j_of_w(w_eff)
        jw_over_w2 = jw / (w_eff * w_eff)

        # coth factor
        T = float(self.temperature_K)
        if T <= 0.0:
            therm = np.ones_like(w_eff)
        else:
            w_q = Q(w_eff, "rad/s")
            T_q = Q(T, "K")
            x_q = (_hbar * w_q) / (2.0 * _kB * T_q)
            x = np.asarray(x_q.to_base_units().magnitude, dtype=float)
            therm = self._coth(x)

        cal = float(calibration)
        if cal <= 0.0:
            return np.zeros_like(w_eff)

        return (float(prefactor) * cal) * om2 * jw_over_w2 * therm

    def _omega_rabi_rad_s(
        self,
        omega_solver: np.ndarray,
        *,
        time_unit_s: float,
        b_polaron: float = 1.0,
    ) -> np.ndarray:
        s = float(time_unit_s)
        if s <= 0.0:
            raise ValueError("time_unit_s must be > 0")
        om = np.asarray(omega_solver, dtype=complex).reshape(-1) / s
        amp = np.sqrt((om.real * om.real) + (om.imag * om.imag))
        return amp * float(b_polaron)

    def omega_star_rad_s(
        self,
        omega_solver: np.ndarray,
        detuning_rad_s: np.ndarray,
        *,
        time_unit_s: float,
        b_polaron: float = 1.0,
        w_floor_rad_s: float = 1.0e8,
    ) -> np.ndarray:
        OmR = self._omega_rabi_rad_s(
            omega_solver, time_unit_s=time_unit_s, b_polaron=b_polaron
        )
        D = np.asarray(detuning_rad_s, dtype=float).reshape(-1)
        w = np.sqrt((D * D) + (OmR * OmR))
        return np.maximum(w, float(w_floor_rad_s))

    def _re_int_exp_iwt_G(
        self,
        w_rad_s: np.ndarray,
        *,
        tau_s: np.ndarray,
        G_tau: np.ndarray,
    ) -> np.ndarray:
        # returns Re \int_0^{tau_max} d tau exp(i w tau) G(tau)
        w = np.asarray(w_rad_s, dtype=float).reshape(-1)
        tau = np.asarray(tau_s, dtype=float).reshape(-1)
        G = np.asarray(G_tau, dtype=complex).reshape(-1)

        dtau = float(tau[1] - tau[0])

        # Broadcasting: (Nt, 1) * (1, Nw) -> (Nt, Nw)
        phase = tau[:, None] * w[None, :]
        epos = np.exp(1j * phase)
        integrand = epos * G[:, None]
        val = np.trapezoid(integrand, tau, axis=0)
        return np.real(val).astype(float)


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
        energy: EnergyStructure,
        mixing: ExcitonMixingParams,
        transitions: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY,
    ):
        self._P = params
        self._tr = transitions
        self._energy = energy
        self._mixing = mixing
        self._cache: dict[str, float] = {}

    # ---------------- state couplings ----------------

    def _delta_prime_eV(self) -> float:
        m = self._mixing
        if m is None:
            return 0.0
        return float(m.delta_prime.to("eV").magnitude)

    def _exciton_eigsplitting_rad_s(self) -> float:
        # DeltaE = sqrt(FSS^2 + (2 delta_prime)^2)
        fss_eV = float(
            (self._energy.X1.to("eV") - self._energy.X2.to("eV")).magnitude
        )
        dp_eV = float(self._delta_prime_eV())
        dE_eV = math.sqrt((fss_eV * fss_eV) + (2.0 * dp_eV) * (2.0 * dp_eV))
        if dE_eV <= 0.0:
            return 0.0
        dE_J = Q(dE_eV, "eV").to("J")
        return float((dE_J / _hbar).to("rad/s").magnitude)

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

        key = f"B:{alpha:.6e}:{wc:.6e}:{T:.6e}:{float(s2):.6e}"
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

    def _phenomenological_rates(self) -> dict[RateKey, QuantityLike]:
        ph = self._P.phenomenological
        out: dict[RateKey, QuantityLike] = {}

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

    def _polaron_exciton_relax_rates(self) -> dict[RateKey, QuantityLike]:
        P = self._P
        if not bool(P.polaron_la.enable_exciton_relaxation):
            return {}

        w = float(self._exciton_eigsplitting_rad_s())
        if w <= 0.0:
            return {}

        # requires X1 and X2 to be distinguishable via couplings
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

        # Decide which directed key corresponds to downhill vs uphill
        e_x1 = float(self._energy.X1.to("eV").magnitude)
        e_x2 = float(self._energy.X2.to("eV").magnitude)

        if e_x1 >= e_x2:
            key_down = RateKey.PH_RELAX_X1_X2
            key_up = RateKey.PH_RELAX_X2_X1
        else:
            key_down = RateKey.PH_RELAX_X2_X1
            key_up = RateKey.PH_RELAX_X1_X2

        out: dict[RateKey, QuantityLike] = {}
        if g_down > 0.0:
            out[key_down] = as_quantity(g_down, "1/s")
        if g_up > 0.0:
            out[key_up] = as_quantity(g_up, "1/s")
        return out

    def _state_energy_eV(self, s: QDState) -> float:
        if s is QDState.G:
            return float(self._energy.G.to("eV").magnitude)
        if s is QDState.X1:
            return float(self._energy.X1.to("eV").magnitude)
        if s is QDState.X2:
            return float(self._energy.X2.to("eV").magnitude)
        if s is QDState.XX:
            return float(self._energy.XX.to("eV").magnitude)
        return 0.0

    def _omega_between_states_rad_s(self, high: QDState, low: QDState) -> float:
        e_hi = self._state_energy_eV(high)
        e_lo = self._state_energy_eV(low)
        de_eV = e_hi - e_lo
        if de_eV <= 0.0:
            return 0.0
        de_J = Q(de_eV, "eV").to("J")
        return float((de_J / _hbar).to("rad/s").magnitude)

    def _pair_relax_rates(
        self,
        *,
        high: QDState,
        low: QDState,
        key_down: RateKey,
        key_up: RateKey,
        scale: float = 1.0,
    ) -> dict[RateKey, QuantityLike]:
        w = float(self._omega_between_states_rad_s(high, low))
        if w <= 0.0:
            return {}

        d = self._phi(high) - self._phi(low)
        s2 = float(d * d)
        if s2 <= 0.0:
            return {}

        Jw = self._j_of_w(w)
        if Jw <= 0.0:
            return {}

        n = self._bose_n(w)
        g_down = float(2.0 * math.pi * s2 * Jw * (n + 1.0)) * float(scale)
        g_up = float(2.0 * math.pi * s2 * Jw * n) * float(scale)

        out: dict[RateKey, QuantityLike] = {}
        if g_down > 0.0:
            out[key_down] = as_quantity(g_down, "1/s")
        if g_up > 0.0:
            out[key_up] = as_quantity(g_up, "1/s")
        return out

    def _polaron_static_relax_rates_all(self) -> dict[RateKey, QuantityLike]:
        P = self._P
        if not bool(P.polaron_la.enable_exciton_relaxation):
            return {}

        # Optional: global scaling knob (add to params if you want)
        scale = 1.0
        ph = getattr(P, "phenomenological", None)
        if ph is not None:
            scale = float(getattr(ph, "gamma_relax_global_scale", 1.0))

        out: dict[RateKey, QuantityLike] = {}

        # X1 <-> X2: choose high/low by energy values
        e_x1 = self._state_energy_eV(QDState.X1)
        e_x2 = self._state_energy_eV(QDState.X2)
        if e_x1 >= e_x2:
            out.update(
                self._pair_relax_rates(
                    high=QDState.X1,
                    low=QDState.X2,
                    key_down=RateKey.PH_RELAX_X1_X2,
                    key_up=RateKey.PH_RELAX_X2_X1,
                    scale=scale,
                )
            )
        else:
            out.update(
                self._pair_relax_rates(
                    high=QDState.X2,
                    low=QDState.X1,
                    key_down=RateKey.PH_RELAX_X2_X1,
                    key_up=RateKey.PH_RELAX_X1_X2,
                    scale=scale,
                )
            )

        # X1 <-> G
        out.update(
            self._pair_relax_rates(
                high=QDState.X1,
                low=QDState.G,
                key_down=RateKey.PH_RELAX_X1_G,
                key_up=RateKey.PH_RELAX_G_X1,
                scale=scale,
            )
        )

        # X2 <-> G
        out.update(
            self._pair_relax_rates(
                high=QDState.X2,
                low=QDState.G,
                key_down=RateKey.PH_RELAX_X2_G,
                key_up=RateKey.PH_RELAX_G_X2,
                scale=scale,
            )
        )

        # XX <-> X1
        out.update(
            self._pair_relax_rates(
                high=QDState.XX,
                low=QDState.X1,
                key_down=RateKey.PH_RELAX_XX_X1,
                key_up=RateKey.PH_RELAX_X1_XX,
                scale=scale,
            )
        )

        # XX <-> X2
        out.update(
            self._pair_relax_rates(
                high=QDState.XX,
                low=QDState.X2,
                key_down=RateKey.PH_RELAX_XX_X2,
                key_up=RateKey.PH_RELAX_X2_XX,
                scale=scale,
            )
        )

        return out

    # ---------------- public compute ----------------

    def compute(self) -> PhononOutputs:
        P = self._P
        if P.kind is PhononModelKind.NONE:
            return PhononOutputs()

        rates: dict[RateKey, QuantityLike] = {}
        rates.update(self._phenomenological_rates())

        bmap: dict[Transition, float] = {}
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

        print(bmap)

        pol = P.polaron_la
        eid = PolaronEIDConfig(
            enabled=bool(
                P.kind is PhononModelKind.POLARON_LA and pol.enable_eid
            ),
            alpha_s2=float(pol.alpha.to("s**2").magnitude),
            omega_c_rad_s=float(pol.omega_c.to("rad/s").magnitude),
            temperature_K=float(P.temperature.to("K").magnitude),
        )

        polaron_rates = None
        if eid.enabled:
            polaron_rates = PolaronDriveRates(
                enabled=True,
                alpha_s2=eid.alpha_s2,
                omega_c_rad_s=eid.omega_c_rad_s,
                temperature_K=eid.temperature_K,
            )

        return PhononOutputs(
            rates=rates, b_polaron=bmap, eid=eid, polaron_rates=polaron_rates
        )
