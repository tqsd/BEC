from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from smef.core.drives.types import DriveSpec
from smef.engine import UnitSystem

from bec.quantum_dot.enums import TransitionPair


@dataclass(frozen=True)
class StirapAdiabaticReport:
    # solver grid (dimensionless) and physical time
    tlist_solver: np.ndarray
    t_phys_s: np.ndarray

    # full curve
    R_t: np.ndarray
    omega_eff: np.ndarray
    mask_active: np.ndarray  # bool mask of where we evaluate "meaningfully"

    # summary on full grid (usually not that meaningful)
    R_max: float
    R_p99: float

    # summary on active window (the one you should cite)
    R_max_active: float
    R_p99_active: float

    # useful debug info
    omega_eff_peak: float
    omega_eff_thresh: float


def _central_diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if y.size != t.size:
        raise ValueError("central_diff: len(y) != len(t)")
    if t.size < 3:
        raise ValueError("central_diff: need at least 3 points")

    dy = np.empty_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    dy[0] = (y[1] - y[0]) / (t[1] - t[0])
    dy[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return dy


def _extract_two_omegas(
    *,
    resolved: Sequence[Any],
    coeffs: Any,
    T: int,
    pair_pump: TransitionPair,
    pair_stokes: TransitionPair,
) -> tuple[np.ndarray, np.ndarray]:
    omega_p = None
    omega_s = None

    for rd in resolved:
        pair = getattr(rd, "transition_key", None)
        key = (rd.drive_id, pair)
        if key not in coeffs.coeffs:
            continue

        om = np.asarray(coeffs.coeffs[key], dtype=np.complex128).reshape(-1)
        if om.size != T:
            raise ValueError(
                "Omega length mismatch for drive_id=%s: %d != %d"
                % (str(rd.drive_id), om.size, T)
            )

        if pair == pair_pump:
            omega_p = om
        elif pair == pair_stokes:
            omega_s = om

    if omega_p is None or omega_s is None:
        raise ValueError(
            "Could not locate both pump/stokes Omegas. "
            "Found pump=%s stokes=%s"
            % (omega_p is not None, omega_s is not None)
        )

    return omega_p, omega_s


def stirap_adiabatic_report_from_drives(
    *,
    qd: Any,
    drives: Sequence[DriveSpec],
    tlist_solver: np.ndarray,
    time_unit_s: float,
    pair_pump: TransitionPair = TransitionPair.G_X1,
    pair_stokes: TransitionPair = TransitionPair.X1_XX,
    # windowing and numerical stability
    active_rel: float = 1e-2,  # active window threshold: 1% of peak Omega_eff
    omega_floor_rel: float = 1e-6,  # floor as fraction of peak Omega_eff
) -> StirapAdiabaticReport:
    """
    STIRAP/DPE adiabaticity check based on the two drive Rabi rates.

    IMPORTANT: we compute dtheta/dt in SOLVER time to match Omega arrays
    (which are produced on the solver grid).
    This avoids unit mismatches.

    We also compute "active-window" stats using Omega_eff >= active_rel * peak.
    Those are the meaningful ones.
    """
    tlist_solver = np.asarray(tlist_solver, dtype=float).reshape(-1)
    T = int(tlist_solver.size)
    if T < 3:
        raise ValueError("tlist_solver must have at least 3 points")

    # --- get Omega arrays (solver units) ---
    units = UnitSystem(time_unit_s=float(time_unit_s))
    bundle = qd.compile_bundle(units=units)

    decoder = getattr(bundle, "drive_decoder", None)
    strength = getattr(bundle, "drive_strength", None)
    decode_ctx = getattr(bundle, "drive_decode", None)

    if decoder is None or strength is None:
        raise ValueError("qd bundle missing drive_decoder/drive_strength")

    if decode_ctx is not None and hasattr(decode_ctx, "with_solver_grid"):
        decode_ctx = decode_ctx.with_solver_grid(
            tlist=tlist_solver, time_unit_s=float(time_unit_s)
        )

    resolved = decoder.decode(drives, ctx=decode_ctx)
    coeffs = strength.compute(
        resolved,
        tlist_solver,
        time_unit_s=float(time_unit_s),
        decode_ctx=decode_ctx,
    )

    omega_p, omega_s = _extract_two_omegas(
        resolved=resolved,
        coeffs=coeffs,
        T=T,
        pair_pump=pair_pump,
        pair_stokes=pair_stokes,
    )

    # --- build theta(t) and R(t) ---
    Op = np.abs(omega_p).astype(float)
    Os = np.abs(omega_s).astype(float)
    omega_eff = np.sqrt(Op * Op + Os * Os)

    omega_eff_peak = float(np.max(omega_eff))
    if not np.isfinite(omega_eff_peak) or omega_eff_peak <= 0.0:
        raise ValueError("Omega_eff peak is non-finite or zero")

    omega_eff_thresh = float(active_rel) * omega_eff_peak
    omega_floor = float(omega_floor_rel) * omega_eff_peak

    # theta uses a floor so Os ~ 0 doesn't explode
    theta = np.arctan2(Op, np.maximum(Os, omega_floor))

    # compute derivative w.r.t SOLVER time
    # solver time variable is just tlist_solver; Omega arrays are defined on it.
    dtheta_dt_solver = _central_diff(theta, tlist_solver)

    # R(t) in solver units: dimensionless / (1/solver_time) -> dimensionless
    R_t = np.abs(dtheta_dt_solver) / np.maximum(omega_eff, omega_floor)

    # active window mask
    mask_active = omega_eff >= omega_eff_thresh
    R_active = R_t[mask_active]

    if R_active.size == 0:
        raise ValueError(
            "Active mask removed all points. "
            "Decrease active_rel (currently %g)." % float(active_rel)
        )

    # physical time for convenience in plots/logs
    t_phys_s = float(time_unit_s) * tlist_solver

    return StirapAdiabaticReport(
        tlist_solver=tlist_solver,
        t_phys_s=t_phys_s,
        R_t=R_t,
        omega_eff=omega_eff,
        mask_active=mask_active,
        R_max=float(np.max(R_t)),
        R_p99=float(np.quantile(R_t, 0.99)),
        R_max_active=float(np.max(R_active)),
        R_p99_active=float(np.quantile(R_active, 0.99)),
        omega_eff_peak=omega_eff_peak,
        omega_eff_thresh=omega_eff_thresh,
    )
