from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from smef.core.units import Q, QuantityLike, c, magnitude

from bec.light.core.polarization import JonesMatrix, JonesState
from bec.light.envelopes.registry import envelope_to_json

from ..field_drive import ClassicalFieldDriveU


def _safe_envelope_type(env: Any) -> str:
    try:
        d = envelope_to_json(env)  # type: ignore[arg-type]
        t = d.get("type")
        if isinstance(t, str):
            return t
    except Exception:
        pass
    return type(env).__name__


def _envelope_params(env: Any) -> dict[str, str]:
    out: dict[str, str] = {}

    for name in ("t0", "sigma"):
        if hasattr(env, name):
            v = getattr(env, name)
            if hasattr(v, "to"):
                out[name] = f"{float(magnitude(v, 's'))} s"
            else:
                out[name] = str(v)

    try:
        d = envelope_to_json(env)  # type: ignore[arg-type]
        for k, v in d.items():
            if k == "type":
                continue
            if isinstance(v, dict) and "value" in v and "unit" in v:
                out[f"json.{k}"] = f"{float(v['value'])} {str(v['unit'])}"
            elif isinstance(v, (int, float, str)):
                out[f"json.{k}"] = str(v)
    except Exception:
        pass

    return out


def _t_phys_from_solver(t_solver: float, time_unit_s: float) -> QuantityLike:
    return Q(float(t_solver) * float(time_unit_s), "s")


def _infer_lambda_from_omega(omega_rad_s: float) -> QuantityLike | None:
    if omega_rad_s <= 0.0:
        return None
    f_hz = omega_rad_s / (2.0 * np.pi)
    lam_m = float(magnitude(c, "m/s")) / f_hz
    return Q(lam_m, "m").to("nm")


@dataclass(frozen=True)
class DriveReportData:
    # Identity
    label: str | None
    drive_type: str
    preferred_kind: str | None

    # Envelope
    envelope_type: str
    envelope_params: dict[str, str]

    # Amplitude
    E0_V_m: float

    # Carrier
    has_carrier: bool
    omega0_rad_s: float | None
    delta_omega_repr: str
    phi0: float | None

    # Polarization
    pol_state_repr: str
    pol_transform_repr: str
    effective_pol: np.ndarray | None

    # Sampled at t_eval (solver)
    time_unit_s: float
    sample_window: tuple[float, float]
    t_eval_solver: float
    t_eval_phys_s: float

    E_env_eval_V_m: float
    omega_L_eval_rad_s: float | None
    omega_L_eval_solver: float | None
    lambda_inferred_nm: float | None

    # Optional sampled curve over solver window
    t_solver: np.ndarray | None
    E_env_curve_V_m: np.ndarray | None


def compute_drive_report_data(
    drive: ClassicalFieldDriveU,
    *,
    time_unit_s: float,
    t_eval_solver: float | None = None,
    sample_window: tuple[float, float] = (0.0, 100.0),
    sample_points: int = 1001,
    include_curve: bool = True,
) -> DriveReportData:
    env = drive.envelope

    # Choose eval time
    t_min, t_max = float(sample_window[0]), float(sample_window[1])
    if t_eval_solver is None:
        t_guess: float | None = None
        if hasattr(env, "t0"):
            try:
                t0_s = float(magnitude(getattr(env, "t0"), "s"))
                t_guess = t0_s / float(time_unit_s)
            except Exception:
                t_guess = None
        t_eval_solver = (
            float(t_guess) if t_guess is not None else (t_min + t_max) / 2.0
        )

    t_eval_phys = _t_phys_from_solver(float(t_eval_solver), float(time_unit_s))

    # Evaluate E(t) and omega(t)
    E_eval = float(drive.E_env_V_m(t_eval_phys))
    omega_q = drive.omega_L_phys(t_eval_phys)
    omega_phys = None if omega_q is None else float(magnitude(omega_q, "rad/s"))
    omega_solver = (
        None if omega_phys is None else float(omega_phys * float(time_unit_s))
    )
    lam_q = (
        None
        if omega_phys is None
        else _infer_lambda_from_omega(float(omega_phys))
    )
    lam_nm = None if lam_q is None else float(magnitude(lam_q, "nm"))

    # Carrier summary
    if drive.carrier is None:
        has_carrier = False
        omega0 = None
        delta_repr = "None"
        phi0 = None
    else:
        has_carrier = True
        omega0 = float(magnitude(drive.carrier.omega0, "rad/s"))
        if callable(drive.carrier.delta_omega):
            delta_repr = "callable"
        else:
            # type: ignore[arg-type]
            delta_repr = f"{
                float(magnitude(drive.carrier.delta_omega, 'rad/s'))} rad/s"
        phi0 = float(drive.carrier.phi0)

    # Polarization reprs
    if drive.pol_state is None:
        pol_state_repr = "None"
    else:
        js: JonesState = drive.pol_state
        a = js.as_array()
        pol_state_repr = (
            f"basis={js.basis} normalize={bool(js.normalize)} "
            f"vec=[{a[0].real:.6g}+{a[0].imag:.6g}j, {a[1].real:.6g}+{a[1].imag:.6g}j]"
        )

    if drive.pol_transform is None:
        pol_transform_repr = "None"
    else:
        jm: JonesMatrix = drive.pol_transform
        J = jm.J
        pol_transform_repr = (
            f"basis={jm.basis} "
            f"J=[[{J[0, 0].real:.6g}+{J[0, 0].imag:.6g}j, {J[0,
                                                             1].real:.6g}+{J[0, 1].imag:.6g}j], "
            f"[{J[1, 0].real:.6g}+{J[1, 0].imag:.6g}j, {J[1,
                                                          1].real:.6g}+{J[1, 1].imag:.6g}j]]"
        )

    eff_pol = drive.effective_pol()

    # Optional curve
    t_arr: np.ndarray | None
    y_arr: np.ndarray | None
    if include_curve:
        t_arr = np.linspace(t_min, t_max, int(sample_points), dtype=float)
        y_arr = np.empty_like(t_arr, dtype=float)
        for i in range(t_arr.size):
            tp = _t_phys_from_solver(float(t_arr[i]), float(time_unit_s))
            y_arr[i] = float(drive.E_env_V_m(tp))
    else:
        t_arr = None
        y_arr = None

    return DriveReportData(
        label=drive.label,
        drive_type=type(drive).__name__,
        preferred_kind=drive.preferred_kind,
        envelope_type=_safe_envelope_type(env),
        envelope_params=_envelope_params(env),
        E0_V_m=float(magnitude(drive.amplitude.E0, "V/m")),
        has_carrier=has_carrier,
        omega0_rad_s=omega0,
        delta_omega_repr=delta_repr,
        phi0=phi0,
        pol_state_repr=pol_state_repr,
        pol_transform_repr=pol_transform_repr,
        effective_pol=eff_pol,
        time_unit_s=float(time_unit_s),
        sample_window=(t_min, t_max),
        t_eval_solver=float(t_eval_solver),
        t_eval_phys_s=float(magnitude(t_eval_phys, "s")),
        E_env_eval_V_m=E_eval,
        omega_L_eval_rad_s=omega_phys,
        omega_L_eval_solver=omega_solver,
        lambda_inferred_nm=lam_nm,
        t_solver=t_arr,
        E_env_curve_V_m=y_arr,
    )
