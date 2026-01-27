from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from bec.units import QuantityLike, magnitude, Q

from bec.light.envelopes.base import TimeLike
from bec.light.envelopes.registry import envelope_to_json

from bec.light.core.polarization import JonesMatrix, JonesState

from .drive import ClassicalFieldDrive
from .compile import compile_drive


def _fmt_float(x: float, *, sig: int = 6) -> str:
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{sig}e}"
    return f"{x:.{sig}f}"


def _fmt_q(q: Optional[QuantityLike], unit: str, sig: int = 6) -> str:
    if q is None:
        return "None"
    return f"{_fmt_float(float(magnitude(q, unit)), sig=sig)} {unit}"


def _header(title: str, width: int = 78) -> str:
    title = f" {title} "
    if len(title) >= width:
        return title
    left = (width - len(title)) // 2
    right = width - len(title) - left
    return ("=" * left) + title + ("=" * right)


def _kv(key: str, val: str, k_width: int = 22) -> str:
    key = key[:k_width]
    return f"{key:<{k_width}} : {val}"


def _safe_envelope_type(env: Any) -> str:
    # Prefer registry json type if serializable
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
    # Best-effort: check common attributes (t0, sigma, fwhm, etc.)
    for name in ("t0", "sigma", "fwhm"):
        if hasattr(env, name):
            v = getattr(env, name)
            # Try time quantities first
            if hasattr(v, "to"):
                out[name] = _fmt_q(v, "s")
            else:
                out[name] = _fmt_float(float(v))
    # If serializable, include json keys
    try:
        d = envelope_to_json(env)  # type: ignore[arg-type]
        for k, v in d.items():
            if k == "type":
                continue
            if isinstance(v, dict) and "value" in v and "unit" in v:
                out[f"json.{k}"] = (
                    f"{_fmt_float(float(v['value']))} {
                    v['unit']}"
                )
            elif isinstance(v, (int, float, str)):
                out[f"json.{k}"] = str(v)
    except Exception:
        pass
    return out


def drive_report(
    drive: ClassicalFieldDrive,
    *,
    time_unit_s: float,
    t0_solver: Optional[float] = None,
    sample_window: Tuple[float, float] = (0.0, 100.0),
    sample_points: int = 1001,
    show_ascii_plot: bool = True,
    plot_width: int = 72,
    plot_height: int = 12,
) -> str:
    """
    Build a textual report for a ClassicalFieldDrive.

    - time_unit_s: solver time unit in seconds (same you pass to compile_drive)
    - t0_solver: where to evaluate "peak" numbers in solver time. If None, uses
      envelope's t0 if present and convertible; otherwise uses midpoint of window.
    - sample_window: (t_min_solver, t_max_solver)
    """
    lines: list[str] = []
    lines.append(_header("ClassicalFieldDrive report"))

    lines.append(_kv("label", str(drive.label)))
    lines.append(_kv("type", type(drive).__name__))

    # Envelope
    env = drive.envelope
    lines.append("")
    lines.append(_header("Envelope", width=78))
    lines.append(_kv("envelope_type", _safe_envelope_type(env)))
    for k, v in _envelope_params(env).items():
        lines.append(_kv(k, v))

    # Amplitude
    lines.append("")
    lines.append(_header("Amplitude", width=78))
    lines.append(_kv("E0", _fmt_q(drive.amplitude.E0, "V/m")))

    # Carrier
    lines.append("")
    lines.append(_header("Carrier", width=78))
    if drive.carrier is None:
        lines.append(_kv("carrier", "None"))
    else:
        car = drive.carrier
        lines.append(_kv("omega0", _fmt_q(car.omega0, "rad/s")))
        if callable(car.delta_omega):
            lines.append(_kv("delta_omega", "callable"))
        else:
            # type: ignore[arg-type]
            lines.append(_kv("delta_omega", _fmt_q(car.delta_omega, "rad/s")))
        lines.append(_kv("phi0", _fmt_float(float(car.phi0))))

    # Polarization
    lines.append("")
    lines.append(_header("Polarization", width=78))
    if drive.pol_state is None:
        lines.append(_kv("pol_state", "None"))
    else:
        js = drive.pol_state
        lines.append(_kv("pol_state.basis", str(js.basis)))
        lines.append(_kv("pol_state.normalize", str(bool(js.normalize))))
        a = js.as_array()
        lines.append(
            _kv(
                "pol_state.vector",
                f"[{_fmt_float(a[0].real)}+{_fmt_float(a[0].imag)}j, "
                f"{_fmt_float(a[1].real)}+{_fmt_float(a[1].imag)}j]",
            )
        )
    if drive.pol_transform is None:
        lines.append(_kv("pol_transform", "None"))
    else:
        jm = drive.pol_transform
        lines.append(_kv("pol_transform.basis", str(jm.basis)))
        J = jm.J
        lines.append(
            _kv(
                "pol_transform.J00",
                f"{_fmt_float(
            J[0, 0].real)}+{_fmt_float(J[0, 0].imag)}j",
            )
        )
        lines.append(
            _kv(
                "pol_transform.J01",
                f"{_fmt_float(
            J[0, 1].real)}+{_fmt_float(J[0, 1].imag)}j",
            )
        )
        lines.append(
            _kv(
                "pol_transform.J10",
                f"{_fmt_float(
            J[1, 0].real)}+{_fmt_float(J[1, 0].imag)}j",
            )
        )
        lines.append(
            _kv(
                "pol_transform.J11",
                f"{_fmt_float(
            J[1, 1].real)}+{_fmt_float(J[1, 1].imag)}j",
            )
        )

    # Compiled summary
    compiled = compile_drive(drive, time_unit_s=float(time_unit_s))

    t_min, t_max = float(sample_window[0]), float(sample_window[1])
    if t0_solver is None:
        # Try to detect envelope.t0 (unitful) -> solver units
        t0_solver_guess = None
        if hasattr(env, "t0"):
            try:
                t0_s = float(magnitude(getattr(env, "t0"), "s"))
                t0_solver_guess = t0_s / float(time_unit_s)
            except Exception:
                t0_solver_guess = None
        t0_solver = (
            float(t0_solver_guess)
            if t0_solver_guess is not None
            else (t_min + t_max) / 2.0
        )

    E_peak = float(compiled.E_env_V_m(t0_solver))
    lines.append("")
    lines.append(_header("Compiled (solver-time)", width=78))
    lines.append(
        _kv("time_unit_s", _fmt_float(float(time_unit_s), sig=6) + " s")
    )
    lines.append(_kv("t_eval_solver", _fmt_float(float(t0_solver))))
    lines.append(_kv("E_env_V_m(t_eval)", _fmt_float(E_peak)))

    if compiled.omega_L_solver is None:
        lines.append(_kv("omega_L_solver", "None"))
    else:
        w_solver = float(compiled.omega_L_solver(t0_solver))
        w_phys = w_solver / float(time_unit_s)
        lines.append(_kv("omega_L_solver", _fmt_float(w_solver)))
        lines.append(_kv("omega_L_phys", _fmt_float(w_phys) + " rad/s"))
        # infer wavelength from omega
        c = 299_792_458.0
        f = w_phys / (2.0 * np.pi)
        lam_nm = (c / f) * 1e9
        lines.append(_kv("lambda_inferred", _fmt_float(lam_nm) + " nm"))

    # ASCII plot of envelope and E(t)
    if show_ascii_plot:
        t = np.linspace(t_min, t_max, int(sample_points), dtype=float)
        y = np.array([compiled.E_env_V_m(tt) for tt in t], dtype=float)
        lines.append("")
        lines.append(_header("ASCII pulse (E_env_V_m)", width=78))
        lines.extend(_ascii_plot(t, y, width=plot_width, height=plot_height))

    return "\n".join(lines)


def _ascii_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: int = 72,
    height: int = 12,
) -> list[str]:
    """
    Simple ASCII plot without external dependencies.
    Downsamples to 'width' columns and draws 'height' rows.
    """
    if x.size == 0:
        return ["(empty)"]

    width = max(10, int(width))
    height = max(4, int(height))

    # Downsample
    idx = np.linspace(0, x.size - 1, width).astype(int)
    xs = x[idx]
    ys = y[idx]

    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    if y_max == y_min:
        y_max = y_min + 1.0

    # Normalize into [0, height-1]
    yn = (ys - y_min) / (y_max - y_min)
    rows = []
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for i in range(width):
        r = int(round((height - 1) - yn[i] * (height - 1)))
        r = max(0, min(height - 1, r))
        grid[r][i] = "*"

    # Add axes labels (top/bottom)
    rows.append(f"y_max={_fmt_float(y_max)}")
    for r in range(height):
        rows.append("".join(grid[r]))
    rows.append(f"y_min={_fmt_float(y_min)}")
    rows.append(
        f"x: [{_fmt_float(float(xs[0]))} .. {
                _fmt_float(float(xs[-1]))}] (solver units)"
    )
    return rows
