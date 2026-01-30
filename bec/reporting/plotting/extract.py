from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from smef.core.units import Q, magnitudes, hbar

from .traces import DriveSeries, QDTraces


_POP_KEYS = ("pop_G", "pop_X1", "pop_X2", "pop_XX")
_OUT_KEYS = ("n_GX_H", "n_GX_V", "n_XX_H", "n_XX_V")


def _as_1d(x: Any, n: int, *, name: str) -> np.ndarray:
    y = np.asarray(x).reshape(-1)
    if int(y.shape[0]) != int(n):
        raise ValueError(f"{name} length {y.shape[0]} but expected {n}")
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


def _drive_label(d: Any, idx: int) -> str:
    lab = getattr(d, "label", None)
    if isinstance(lab, str) and lab.strip():
        return lab
    return f"drive_{idx}"


def _normalize_drives(
    drives: Optional[Union[Any, Sequence[Optional[Any]]]],
) -> Tuple[Any, ...]:
    """
    Accept:
      - None
      - single drive object
      - list/tuple of drives (can include None)
    Return a tuple of non-None drive objects.
    """
    if drives is None:
        return ()

    if isinstance(drives, (list, tuple)):
        out = [d for d in drives if d is not None]
        return tuple(out)

    # Single drive object
    return (drives,)


def _maybe_eval_delta_omega_rad_s(
    drive_obj: Any, t_s: np.ndarray
) -> Optional[np.ndarray]:
    """
    Best-effort extraction of delta_omega(t) in rad/s if the drive exposes it.

    We try:
      - drive.carrier.delta_omega_phys(t)
      - drive.carrier.delta_omega(t)
      - drive.carrier.delta_omega.fn(t)
      - drive.carrier.delta_omega.eval(t)
    """
    carrier = getattr(drive_obj, "carrier", None)
    if carrier is None:
        return None

    # Case 1: method
    fn = getattr(carrier, "delta_omega_phys", None)
    if callable(fn):
        try:
            vals = [fn(float(ts)) for ts in t_s]
            return np.asarray(
                magnitudes(Q(vals, "rad/s"), "rad/s"), dtype=float
            )
        except Exception:
            return None

    # Case 2: attribute callable
    dw = getattr(carrier, "delta_omega", None)
    if callable(dw):
        try:
            vals = [dw(float(ts)) for ts in t_s]
            return np.asarray(
                magnitudes(Q(vals, "rad/s"), "rad/s"), dtype=float
            )
        except Exception:
            return None

    # Case 3: object with .fn or .eval
    if dw is not None:
        fn2 = getattr(dw, "fn", None)
        if callable(fn2):
            try:
                vals = [fn2(float(ts)) for ts in t_s]
                return np.asarray(
                    magnitudes(Q(vals, "rad/s"), "rad/s"), dtype=float
                )
            except Exception:
                return None

        ev = getattr(dw, "eval", None)
        if callable(ev):
            try:
                vals = [ev(float(ts)) for ts in t_s]
                return np.asarray(
                    magnitudes(Q(vals, "rad/s"), "rad/s"), dtype=float
                )
            except Exception:
                return None

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
    pops: dict[str, np.ndarray] = {}
    for k in pop_keys:
        if k in expect:
            arr = _as_1d(expect[k], n_full, name=k)[sl]
            pops[str(k)] = np.asarray(np.real(arr), dtype=float)

    # --- Outputs ---
    outputs: dict[str, np.ndarray] = {}
    for k in out_keys:
        if k in expect:
            arr = _as_1d(expect[k], n_full, name=k)[sl]
            outputs[str(k)] = np.asarray(np.real(arr), dtype=float)

    # --- Coherences ---
    coherences: dict[str, np.ndarray] = {}
    for k, arr in expect.items():
        if isinstance(k, str) and k.startswith(coherence_prefix):
            coherences[k] = np.asarray(
                _as_1d(arr, n_full, name=k)[sl], dtype=complex
            )

    # --- Extra ---
    extra: dict[str, np.ndarray] = {}
    if extra_keys is not None:
        for k in extra_keys:
            if k in expect:
                extra[str(k)] = np.asarray(
                    _as_1d(expect[k], n_full, name=str(k))[sl]
                )

    # --- Drives (multi-drive overlay support) ---
    drives_in = _normalize_drives(drives)
    drive_series: list[DriveSeries] = []

    # Dipole magnitude for Omega overlay (optional)
    mu = None
    if qd is not None:
        mu = getattr(getattr(qd, "dipoles", None), "mu_default", None)

    for i, d in enumerate(drives_in):
        lab = _drive_label(d, i)

        E = None
        wL = None
        dw = None
        Om = None

        # Envelope E(t)
        fn_E = getattr(d, "E_env_V_m", None)
        if callable(fn_E):
            try:
                E = np.asarray(
                    [float(fn_E(float(ts))) for ts in t_s], dtype=float
                )
            except Exception:
                E = None

        # Laser omega_L(t)
        fn_wL = getattr(d, "omega_L_rad_s", None)
        if callable(fn_wL):
            try:
                tmp = [fn_wL(float(ts)) for ts in t_s]
                if all(x is not None for x in tmp):
                    wL = np.asarray([float(x) for x in tmp], dtype=float)
            except Exception:
                wL = None

        # Chirp/delta_omega(t)
        try:
            dw = _maybe_eval_delta_omega_rad_s(d, t_s)
        except Exception:
            dw = None

        # Omega(t) = mu * E(t) / hbar
        if mu is not None and E is not None:
            try:
                Om_q = (mu * Q(E, "V/m")) / hbar
                Om = np.asarray(magnitudes(Om_q, "rad/s"), dtype=float)
            except Exception:
                Om = None

        if E is not None or wL is not None or dw is not None or Om is not None:
            drive_series.append(
                DriveSeries(
                    label=lab,
                    t_s=t_s,
                    E_env_V_m=E,
                    omega_L_rad_s=wL,
                    delta_omega_rad_s=dw,
                    Omega_rad_s=Om,
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
        extra=extra,
        meta=dict(getattr(res, "meta", {}) or {}),
    )
