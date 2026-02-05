from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import math

import numpy as np

from qsi.state import State, StateProp

# BEC/QD-side
from bec.quantum_dot.enums import TransitionPair
from bec.quantum_dot.transitions import RateKey


_C_M_S = 299792458.0


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _omega_to_lambda_nm(omega_rad_s: float) -> float:
    w = float(omega_rad_s)
    if not np.isfinite(w) or w <= 0.0:
        return float("nan")
    lam_m = (2.0 * math.pi * _C_M_S) / w
    return float(lam_m * 1.0e9)


def _rate_1_s(rates: Mapping[RateKey, Any], key: RateKey) -> Optional[float]:
    """
    Extract a non-negative rate in 1/s from qd.rates, if present.
    """
    if key not in rates:
        return None
    v = rates[key]
    # v should be a QuantityLike; but allow raw floats too
    try:
        if hasattr(v, "to"):
            r = float(v.to("1/s").magnitude)
        else:
            r = float(v)
    except Exception:
        return None
    if not np.isfinite(r) or r < 0.0:
        return None
    return float(r)


def _bandwidth_hz_from_gamma(gamma_1_s: float) -> float:
    g = float(gamma_1_s)
    if not np.isfinite(g) or g <= 0.0:
        return float("nan")
    return float(g / (2.0 * math.pi))


def stateprops_photons_from_qd(
    qd: Any,
    *,
    trunc_per_pol: int,
    default_bandwidth_Hz: float = 1.0,
    wavelength_override_nm: Optional[Mapping[str, float]] = None,
) -> list[StateProp]:
    """
    Build QSI light-mode StateProps for the photonic subsystem in *QDModes order*:
      GX_H, GX_V, XX_H, XX_V

    Wavelengths are inferred from qd.derived_view.omega_ref_rad_s(...) using:
        lambda = 2*pi*c / omega

    Bandwidths are inferred from radiative decay rates when available:
        bw_Hz = gamma / (2*pi)

    Args
    ----
    qd:
        QuantumDot (must expose qd.derived_view and qd.rates).
    trunc_per_pol:
        Fock truncation per polarization mode (e.g. 2).
    default_bandwidth_Hz:
        Fallback bandwidth used when rates are unavailable.
    wavelength_override_nm:
        Optional map with keys {"GX","XX"} to override inferred wavelengths.

    Returns
    -------
    list[StateProp]:
        [GX_H, GX_V, XX_H, XX_V]
    """
    dv = getattr(qd, "derived_view", None)
    if dv is None:
        raise TypeError("qd must have .derived_view")

    rates = getattr(qd, "rates", None)
    if rates is None:
        raise TypeError("qd must have .rates")

    # --- wavelengths (nm) ---
    if wavelength_override_nm is None:
        wavelength_override_nm = {}

    # GX band photon energy is X -> G, but omega_ref is symmetric in endpoints.
    w_gx_1 = float(dv.omega_ref_rad_s(TransitionPair.G_X1))
    w_gx_2 = float(dv.omega_ref_rad_s(TransitionPair.G_X2))
    lam_gx_nm = _safe_float(wavelength_override_nm.get("GX", float("nan")))
    if not np.isfinite(lam_gx_nm):
        lam_gx_nm = float(
            np.nanmean(
                [_omega_to_lambda_nm(w_gx_1), _omega_to_lambda_nm(w_gx_2)]
            )
        )

    # XX band photon energy is XX -> X, use X -> XX pair (symmetric).
    w_xx_1 = float(dv.omega_ref_rad_s(TransitionPair.X1_XX))
    w_xx_2 = float(dv.omega_ref_rad_s(TransitionPair.X2_XX))
    lam_xx_nm = _safe_float(wavelength_override_nm.get("XX", float("nan")))
    if not np.isfinite(lam_xx_nm):
        lam_xx_nm = float(
            np.nanmean(
                [_omega_to_lambda_nm(w_xx_1), _omega_to_lambda_nm(w_xx_2)]
            )
        )

    # --- bandwidths (Hz) from radiative decay rates ---
    g_x1 = _rate_1_s(rates, RateKey.RAD_X1_G)
    g_x2 = _rate_1_s(rates, RateKey.RAD_X2_G)
    g_gx = float(
        np.nanmean(
            [
                g_x1 if g_x1 is not None else float("nan"),
                g_x2 if g_x2 is not None else float("nan"),
            ]
        )
    )

    if not np.isfinite(g_gx) or g_gx <= 0.0:
        bw_gx_hz = float(default_bandwidth_Hz)
    else:
        bw_gx_hz = _bandwidth_hz_from_gamma(g_gx)

    g_xx1 = _rate_1_s(rates, RateKey.RAD_XX_X1)
    g_xx2 = _rate_1_s(rates, RateKey.RAD_XX_X2)
    g_xx = float(
        np.nanmean(
            [
                g_xx1 if g_xx1 is not None else float("nan"),
                g_xx2 if g_xx2 is not None else float("nan"),
            ]
        )
    )

    if not np.isfinite(g_xx) or g_xx <= 0.0:
        bw_xx_hz = float(default_bandwidth_Hz)
    else:
        bw_xx_hz = _bandwidth_hz_from_gamma(g_xx)

    # --- build props in QDModes photonic order ---
    t = int(trunc_per_pol)
    if t < 1:
        raise ValueError("trunc_per_pol must be >= 1")

    props: list[StateProp] = [
        StateProp(
            state_type="light",
            truncation=t,
            wavelength=float(lam_gx_nm),
            polarization="H",
            bandwidth=float(bw_gx_hz),
        ),
        StateProp(
            state_type="light",
            truncation=t,
            wavelength=float(lam_gx_nm),
            polarization="V",
            bandwidth=float(bw_gx_hz),
        ),
        StateProp(
            state_type="light",
            truncation=t,
            wavelength=float(lam_xx_nm),
            polarization="H",
            bandwidth=float(bw_xx_hz),
        ),
        StateProp(
            state_type="light",
            truncation=t,
            wavelength=float(lam_xx_nm),
            polarization="V",
            bandwidth=float(bw_xx_hz),
        ),
    ]
    return props


def apply_prepare_from_scalar(
    kraus_ops: Sequence[np.ndarray],  # list of NumPy arrays
    layout: List[StateProp],
    *,
    normalize: bool = True,
) -> State:
    r"""
    Construct a new State with the given factor layout and set:

    .. math::
        \rho_{out} = \sum_i K_i K_i^\dagger
    for a prepare-from-scalar channel (each K_i : C → H_out).

    Accepts K_i shapes:
      - (N_out,)      → reshaped to (N_out, 1)
      - (N_out, 1)    → used as-is
    """
    if not layout:
        raise ValueError("layout must contain at least one StateProp.")
    if not kraus_ops:
        raise ValueError("kraus_ops must be non-empty.")

    out = State(layout[0])
    for p in layout[1:]:
        out.join(State(p))
    N_out = out.dimensions

    rho = np.zeros((N_out, N_out), dtype=complex)
    for K in kraus_ops:
        K = np.asarray(K, dtype=complex)
        if K.ndim == 1:
            K = K.reshape(N_out, 1)
        if K.shape != (N_out, 1):
            raise ValueError(
                f"Kraus op has shape {K.shape}, expected ({N_out}, 1) or ({
                    N_out},)."
            )
        rho += K @ K.conj().T

    if normalize:
        tr = float(np.trace(rho).real)
        if tr > 0.0:
            rho /= tr

    out.state = rho
    return out
