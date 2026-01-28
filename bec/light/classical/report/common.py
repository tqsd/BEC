from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from smef.core.units import QuantityLike, Q, c, magnitude

from bec.light.envelopes.registry import envelope_to_json
from bec.light.core.polarization import JonesMatrix, JonesState

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


def _envelope_params(env: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}

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


def _infer_lambda_from_omega(omega_rad_s: float) -> Optional[QuantityLike]:
    if omega_rad_s <= 0.0:
        return None
    f_hz = omega_rad_s / (2.0 * np.pi)
    lam_m = float(magnitude(c, "m/s")) / f_hz
    return Q(lam_m, "m").to("nm")


@dataclass(frozen=True)
class DriveReportData:
    # Identity
    label: Optional[str]
    drive_type: str
    preferred_kind: Optional[str]

    # Envelope
    envelope_type: str
    envelope_params: Dict[str, str]

    # Amplitude
    E0_V_m: float

    # Carrier
    has_carrier: bool
    omega0_rad_s: Optional[float]
    delta_omega_repr: str
    phi0: Optional[float]

    # Polarization
    pol_state_repr: str
    pol_transform_repr: str
    effective_pol: Optional[np.ndarray]

    # Sampled at t_eval (solver)
    time_unit_s: float
    sample_window: Tuple[float, float]
    t_eval_solver: float
    t_eval_phys_s: float

    E_env_eval_V_m: float
    omega_L_eval_rad_s: Optional[float]
    omega_L_eval_solver: Optional[float]
    lambda_inferred_nm: Optional[float]

    # Optional sampled curve over solver window
    t_solver: Optional[np.ndarray]
    E_env_curve_V_m: Optional[np.ndarray]

    # Peak within sample window (computed from sampled curve if available)
    t_peak_solver: Optional[float]
    t_peak_phys_s: Optional[float]
    E_env_peak_V_m: Optional[float]


def compute_drive_report_data(
    drive: ClassicalFieldDriveU,
    *,
    time_unit_s: float,
    t_eval_solver: Optional[float] = None,
    sample_window: Tuple[float, float] = (0.0, 100.0),
    sample_points: int = 1001,
    include_curve: bool = True,
) -> DriveReportData:
    env = drive.envelope

    # Choose eval time
    t_min, t_max = float(sample_window[0]), float(sample_window[1])
    if t_eval_solver is None:
        t_guess: Optional[float] = None
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
    t_arr: Optional[np.ndarray]
    y_arr: Optional[np.ndarray]
    if include_curve:
        t_arr = np.linspace(t_min, t_max, int(sample_points), dtype=float)
        y_arr = np.empty_like(t_arr, dtype=float)
        for i in range(t_arr.size):
            tp = _t_phys_from_solver(float(t_arr[i]), float(time_unit_s))
            y_arr[i] = float(drive.E_env_V_m(tp))
    else:
        t_arr = None
        y_arr = None

    t_peak_solver: Optional[float] = None
    t_peak_phys_s: Optional[float] = None
    E_peak: Optional[float] = None

    if t_arr is not None and y_arr is not None and y_arr.size > 0:
        i_max = int(np.argmax(y_arr))
        t_peak_solver = float(t_arr[i_max])
        t_peak_phys = _t_phys_from_solver(t_peak_solver, float(time_unit_s))
        t_peak_phys_s = float(magnitude(t_peak_phys, "s"))
        E_peak = float(y_arr[i_max])

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
        t_peak_solver=t_peak_solver,
        t_peak_phys_s=t_peak_phys_s,
        E_env_peak_V_m=E_peak,
    )
