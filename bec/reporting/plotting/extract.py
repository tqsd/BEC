from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from smef.core.units import Q, hbar, magnitude

from .traces import DriveSeries, QDTraces


_POP_KEYS = ("pop_G", "pop_X1", "pop_X2", "pop_XX")
_OUT_KEYS = ("n_GX_H", "n_GX_V", "n_XX_H", "n_XX_V")

# h*c in eV*nm (good precision for plotting)
HC_EV_NM = 1239.841984
C_M_S = 299792458.0


def _as_1d(x: Any, n: int, *, name: str) -> np.ndarray:
    y = np.asarray(x).reshape(-1)
    if int(y.shape[0]) != int(n):
        raise ValueError(
            "%s length %d but expected %d" % (name, int(y.shape[0]), int(n))
        )
    return y


def _slice_window(
    t_s: np.ndarray, window_s: Optional[Tuple[float, float]]
) -> slice:
    if window_s is None:
        return slice(None)
    t0 = float(window_s[0])
    t1 = float(window_s[1])
    if t1 < t0:
        t0, t1 = t1, t0
    i0 = int(np.searchsorted(t_s, t0, side="left"))
    i1 = int(np.searchsorted(t_s, t1, side="right"))
    return slice(i0, i1)


def _normalize_drives(
    drives: Optional[Union[Any, Sequence[Optional[Any]]]],
) -> Tuple[Any, ...]:
    if drives is None:
        return ()
    if isinstance(drives, (list, tuple)):
        return tuple(d for d in drives if d is not None)
    return (drives,)


def _energy_eV(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x.to("eV").magnitude)
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def _wavelength_nm_from_eV(E_eV: Optional[float]) -> Optional[float]:
    if E_eV is None:
        return None
    E = float(E_eV)
    if not np.isfinite(E) or E <= 0.0:
        return None
    return float(HC_EV_NM / E)


def _output_wavelengths_from_qd(
    qd: Any,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute emission wavelengths from the level energies in qd.energy.

    Keys:
      GX_X1, GX_X2, GX_center
      XX_X1, XX_X2, XX_center
    """
    wl_nm: Dict[str, float] = {}
    E_eV: Dict[str, float] = {}

    energy = getattr(qd, "energy", None)
    if energy is None:
        return wl_nm, E_eV

    E_G = _energy_eV(getattr(energy, "G", None))
    if E_G is None:
        E_G = 0.0

    E_X1 = _energy_eV(getattr(energy, "X1", None))
    E_X2 = _energy_eV(getattr(energy, "X2", None))
    E_XX = _energy_eV(getattr(energy, "XX", None))

    E_EX = _energy_eV(getattr(energy, "exciton_center", None))
    if E_EX is None and (E_X1 is not None) and (E_X2 is not None):
        E_EX = 0.5 * (E_X1 + E_X2)

    # X -> G
    if E_X1 is not None:
        e = float(E_X1 - E_G)
        E_eV["GX_X1"] = e
        lam = _wavelength_nm_from_eV(e)
        if lam is not None:
            wl_nm["GX_X1"] = lam

    if E_X2 is not None:
        e = float(E_X2 - E_G)
        E_eV["GX_X2"] = e
        lam = _wavelength_nm_from_eV(e)
        if lam is not None:
            wl_nm["GX_X2"] = lam

    if E_EX is not None:
        e = float(E_EX - E_G)
        E_eV["GX_center"] = e
        lam = _wavelength_nm_from_eV(e)
        if lam is not None:
            wl_nm["GX_center"] = lam

    # XX -> X
    if (E_XX is not None) and (E_X1 is not None):
        e = float(E_XX - E_X1)
        E_eV["XX_X1"] = e
        lam = _wavelength_nm_from_eV(e)
        if lam is not None:
            wl_nm["XX_X1"] = lam

    if (E_XX is not None) and (E_X2 is not None):
        e = float(E_XX - E_X2)
        E_eV["XX_X2"] = e
        lam = _wavelength_nm_from_eV(e)
        if lam is not None:
            wl_nm["XX_X2"] = lam

    if (E_XX is not None) and (E_EX is not None):
        e = float(E_XX - E_EX)
        E_eV["XX_center"] = e
        lam = _wavelength_nm_from_eV(e)
        if lam is not None:
            wl_nm["XX_center"] = lam

    return wl_nm, E_eV


def _carrier_omega0_rad_s(drive_obj: Any) -> Optional[float]:
    carrier = getattr(drive_obj, "carrier", None)
    if carrier is None:
        return None
    w0 = getattr(carrier, "omega0", None)
    if w0 is None:
        return None
    try:
        return float(magnitude(w0, "rad/s"))
    except Exception:
        try:
            return float(w0)
        except Exception:
            return None


def _carrier_delta_omega_rad_s(
    drive_obj: Any, t_s: np.ndarray
) -> Optional[np.ndarray]:
    """
    Sample carrier.delta_omega on the provided physical time grid t_s [s].

    Works for:
      - constant QuantityLike (or numeric)
      - callable OmegaFn: QuantityLike time -> QuantityLike rad/s
    """
    carrier = getattr(drive_obj, "carrier", None)
    if carrier is None:
        return None

    d = getattr(carrier, "delta_omega", None)
    if d is None:
        return None

    # Callable OmegaFn
    if callable(d):
        try:
            out = np.empty(int(t_s.size), dtype=float)
            for i, ts in enumerate(t_s):
                dw_q = d(Q(float(ts), "s"))
                out[i] = float(magnitude(dw_q, "rad/s"))
            return out
        except Exception:
            return None

    # Constant QuantityLike / numeric
    try:
        dw0 = float(magnitude(d, "rad/s"))
    except Exception:
        try:
            dw0 = float(d)
        except Exception:
            return None

    return np.full(int(t_s.size), float(dw0), dtype=float)


def _drive_wavelength_nm_from_omega0(
    w0_rad_s: Optional[float],
) -> Optional[float]:
    if w0_rad_s is None:
        return None
    w0 = float(w0_rad_s)
    if not np.isfinite(w0) or w0 <= 0.0:
        return None
    lam_m = 2.0 * np.pi * C_M_S / w0
    return float(lam_m * 1e9)


def _drive_label(
    drive_obj: Any, idx: int, wavelength_nm: Optional[float]
) -> str:
    lab = getattr(drive_obj, "label", None)
    if isinstance(lab, str) and lab.strip():
        base = lab.strip()
    else:
        base = "drive_%d" % int(idx)

    if wavelength_nm is None:
        return base
    return "%s (%.0f nm)" % (base, float(wavelength_nm))


def _sample_callable_over_time(
    fn: Any, t_s: np.ndarray
) -> Optional[np.ndarray]:
    """
    Sample a callable that accepts TimeLike; we pass float seconds.

    Returns None on any failure.
    """
    if not callable(fn):
        return None
    try:
        vals = [fn(float(ts)) for ts in t_s]
        # allow optional None returns
        if any(v is None for v in vals):
            return None
        return np.asarray([float(v) for v in vals], dtype=float)
    except Exception:
        return None


def extract_qd_traces(
    res: Any,
    *,
    units: Any,
    drives: Optional[Union[Any, Sequence[Optional[Any]]]] = None,
    qd: Optional[Any] = None,
    pop_keys: Sequence[str] = _POP_KEYS,
    out_keys: Sequence[str] = _OUT_KEYS,
    coherence_prefix: str = "coh_",
    extra_keys: Optional[Iterable[str]] = None,
    window_s: Optional[Tuple[float, float]] = None,
) -> QDTraces:
    # --- Time axis ---
    t_solver_full = np.asarray(
        getattr(res, "tlist", None), dtype=float
    ).reshape(-1)
    if t_solver_full.size == 0:
        raise ValueError("res.tlist must be non-empty")

    time_unit_s = float(getattr(units, "time_unit_s", None))
    t_s_full = t_solver_full * time_unit_s

    sl = _slice_window(t_s_full, window_s)
    t_solver = t_solver_full[sl]
    t_s = t_s_full[sl]

    # --- Expectation dictionary ---
    expect = getattr(res, "expect", None)
    if not isinstance(expect, Mapping):
        raise ValueError("res.expect must be a mapping")

    n_full = int(t_s_full.shape[0])

    # --- Populations ---
    pops: Dict[str, np.ndarray] = {}
    for k in pop_keys:
        if k in expect:
            arr = _as_1d(expect[k], n_full, name=str(k))[sl]
            pops[str(k)] = np.asarray(np.real(arr), dtype=float)

    # --- Outputs ---
    outputs: Dict[str, np.ndarray] = {}
    for k in out_keys:
        if k in expect:
            arr = _as_1d(expect[k], n_full, name=str(k))[sl]
            outputs[str(k)] = np.asarray(np.real(arr), dtype=float)

    # --- Coherences ---
    coherences: Dict[str, np.ndarray] = {}
    for k, arr in expect.items():
        if isinstance(k, str) and k.startswith(coherence_prefix):
            coherences[k] = np.asarray(
                _as_1d(arr, n_full, name=k)[sl], dtype=complex
            )

    # --- Extra ---
    extra: Dict[str, np.ndarray] = {}
    if extra_keys is not None:
        for k in extra_keys:
            if k in expect:
                extra[str(k)] = np.asarray(
                    _as_1d(expect[k], n_full, name=str(k))[sl]
                )

    # --- Output wavelength metadata (first-class) ---
    output_wavelengths_nm: Dict[str, float] = {}
    output_transition_energies_eV: Dict[str, float] = {}
    if qd is not None:
        output_wavelengths_nm, output_transition_energies_eV = (
            _output_wavelengths_from_qd(qd)
        )

    # --- Drives ---
    drives_in = _normalize_drives(drives)
    drive_series: List[DriveSeries] = []
    drive_wavelengths_nm: Dict[str, float] = {}

    # Dipole magnitude for Omega overlay (optional)
    mu = None
    if qd is not None:
        mu = getattr(getattr(qd, "dipoles", None), "mu_default", None)

    for i, d in enumerate(drives_in):
        w0 = _carrier_omega0_rad_s(d)
        wl_nm = _drive_wavelength_nm_from_omega0(w0)

        lab = _drive_label(d, i, wl_nm)
        if wl_nm is not None:
            drive_wavelengths_nm[lab] = float(wl_nm)

        # Envelope E(t) in V/m (float)
        E = _sample_callable_over_time(getattr(d, "E_env_V_m", None), t_s)

        # Instantaneous omega_L(t) in rad/s (float) if available as callable
        wL = _sample_callable_over_time(getattr(d, "omega_L_rad_s", None), t_s)

        # delta_omega(t) in rad/s from Carrier (constant or callable OmegaFn)
        dw = _carrier_delta_omega_rad_s(d, t_s)

        # Omega(t) = mu * E(t) / hbar (optional)
        Om = None
        if mu is not None and E is not None:
            try:
                Om_q = (mu * Q(E, "V/m")) / hbar
                Om = np.asarray(
                    [float(magnitude(x, "rad/s")) for x in Om_q], dtype=float
                )
            except Exception:
                Om = None

        if (
            (E is not None)
            or (wL is not None)
            or (dw is not None)
            or (Om is not None)
        ):
            drive_series.append(
                DriveSeries(
                    label=lab,
                    t_s=t_s,
                    E_env_V_m=E,
                    omega_L_rad_s=wL,
                    delta_omega_rad_s=dw,
                    Omega_rad_s=Om,
                    wavelength_nm=wl_nm,
                )
            )

    return QDTraces(
        t_solver=t_solver,
        t_s=t_s,
        time_unit_s=float(time_unit_s),
        pops=pops,
        outputs=outputs,
        coherences=coherences,
        drives=tuple(drive_series),
        output_wavelengths_nm=output_wavelengths_nm,
        output_transition_energies_eV=output_transition_energies_eV,
        drive_wavelengths_nm=drive_wavelengths_nm,
        extra=extra,
        meta=dict(getattr(res, "meta", {}) or {}),
    )
