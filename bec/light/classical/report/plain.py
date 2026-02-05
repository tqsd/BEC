from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .common import DriveReportData


def _fmt_float(x: float, sig: int = 6) -> str:
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{sig}e}"
    return f"{x:.{sig}f}"


def _header(title: str, width: int = 78) -> str:
    t = f" {title} "
    if len(t) >= width:
        return t
    left = (width - len(t)) // 2
    right = width - len(t) - left
    return ("=" * left) + t + ("=" * right)


def _kv(key: str, val: str, k_width: int = 22) -> str:
    key2 = key[:k_width]
    return f"{key2:<{k_width}} : {val}"


def _ascii_plot(
    x: np.ndarray, y: np.ndarray, width: int = 72, height: int = 12
) -> Sequence[str]:
    if x.size == 0:
        return ["(empty)"]

    width = max(10, int(width))
    height = max(4, int(height))

    idx = np.linspace(0, x.size - 1, width).astype(int)
    xs = x[idx]
    ys = y[idx]

    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    if y_max == y_min:
        y_max = y_min + 1.0

    yn = (ys - y_min) / (y_max - y_min)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for i in range(width):
        r = int(round((height - 1) - yn[i] * (height - 1)))
        r = max(0, min(height - 1, r))
        grid[r][i] = "*"

    rows: list[str] = []
    rows.append(f"y_max={_fmt_float(y_max)}")
    for r in range(height):
        rows.append("".join(grid[r]))
    rows.append(f"y_min={_fmt_float(y_min)}")
    rows.append(
        f"x: [{_fmt_float(float(xs[0]))} .. {
            _fmt_float(float(xs[-1]))}] (solver units)"
    )
    return rows


def render_plain(
    rep: DriveReportData,
    *,
    show_ascii_plot: bool = True,
    plot_width: int = 72,
    plot_height: int = 12,
) -> str:
    lines: list[str] = []
    lines.append(_header("ClassicalFieldDrive report"))

    lines.append(_kv("label", str(rep.label)))
    lines.append(_kv("type", rep.drive_type))
    if rep.preferred_kind is not None:
        lines.append(_kv("preferred_kind", str(rep.preferred_kind)))

    lines.append("")
    lines.append(_header("Envelope"))
    lines.append(_kv("envelope_type", rep.envelope_type))
    for k, v in rep.envelope_params.items():
        lines.append(_kv(k, v))

    lines.append("")
    lines.append(_header("Amplitude"))
    lines.append(_kv("E0", _fmt_float(rep.E0_V_m) + " V/m"))

    lines.append("")
    lines.append(_header("Carrier"))
    if not rep.has_carrier:
        lines.append(_kv("carrier", "None"))
    else:
        lines.append(
            _kv("omega0", _fmt_float(float(rep.omega0_rad_s)) + " rad/s")
        )
        lines.append(_kv("delta_omega", rep.delta_omega_repr))
        lines.append(_kv("phi0", _fmt_float(float(rep.phi0))))

    lines.append("")
    lines.append(_header("Polarization"))
    lines.append(_kv("pol_state", rep.pol_state_repr))
    lines.append(_kv("pol_transform", rep.pol_transform_repr))

    lines.append("")
    lines.append(_header("Sampled at solver-time"))
    lines.append(_kv("time_unit_s", _fmt_float(rep.time_unit_s) + " s"))
    lines.append(_kv("t_eval_solver", _fmt_float(rep.t_eval_solver)))
    lines.append(_kv("t_eval_phys_s", _fmt_float(rep.t_eval_phys_s) + " s"))
    lines.append(_kv("E_env_V_m(t_eval)", _fmt_float(rep.E_env_eval_V_m)))
    if (
        rep.E_env_peak_V_m is not None
        and rep.t_peak_solver is not None
        and rep.t_peak_phys_s is not None
    ):
        lines.append(_kv("t_peak_solver", _fmt_float(rep.t_peak_solver)))
        lines.append(_kv("t_peak_phys_s", _fmt_float(rep.t_peak_phys_s) + " s"))
        lines.append(_kv("E_env_peak_V_m", _fmt_float(rep.E_env_peak_V_m)))

    if rep.omega_L_eval_rad_s is None:
        lines.append(_kv("omega_L_phys", "None"))
    else:
        lines.append(
            _kv("omega_L_solver", _fmt_float(float(rep.omega_L_eval_solver)))
        )
        lines.append(
            _kv(
                "omega_L_phys",
                _fmt_float(float(rep.omega_L_eval_rad_s)) + " rad/s",
            )
        )
        lines.append(
            _kv(
                "lambda_inferred",
                (
                    "None"
                    if rep.lambda_inferred_nm is None
                    else _fmt_float(rep.lambda_inferred_nm) + " nm"
                ),
            )
        )

    if (
        show_ascii_plot
        and rep.t_solver is not None
        and rep.E_env_curve_V_m is not None
    ):
        lines.append("")
        lines.append(_header("ASCII pulse (E_env_V_m)"))
        lines.extend(
            _ascii_plot(
                rep.t_solver,
                rep.E_env_curve_V_m,
                width=plot_width,
                height=plot_height,
            )
        )

    return "\n".join(lines)
