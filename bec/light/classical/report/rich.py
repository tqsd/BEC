from __future__ import annotations

from .common import DriveReportData
from .plain import render_plain


def render_rich(rep: DriveReportData, *, show_ascii_plot: bool = True) -> str:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except Exception:
        return render_plain(rep, show_ascii_plot=show_ascii_plot)

    console = Console(record=True, width=96)

    def kv_table(title: str) -> Table:
        t = Table(title=title, box=box.SQUARE, show_header=False)
        t.add_column("k", style="bold", width=18)
        t.add_column("v")
        return t

    t0 = kv_table("ClassicalFieldDrive report")
    t0.add_row("label", str(rep.label))
    t0.add_row("type", rep.drive_type)
    if rep.preferred_kind is not None:
        t0.add_row("preferred_kind", str(rep.preferred_kind))
    console.print(t0)

    te = kv_table("Envelope")
    te.add_row("envelope_type", rep.envelope_type)
    for k, v in rep.envelope_params.items():
        te.add_row(k, v)
    console.print(te)

    ta = kv_table("Amplitude")
    ta.add_row("E0", f"{rep.E0_V_m:.6g} V/m")
    console.print(ta)

    tc = kv_table("Carrier")
    if not rep.has_carrier:
        tc.add_row("carrier", "None")
    else:
        tc.add_row("omega0", f"{float(rep.omega0_rad_s):.6g} rad/s")
        tc.add_row("delta_omega", rep.delta_omega_repr)
        tc.add_row("phi0", f"{float(rep.phi0):.6g}")
    console.print(tc)

    tp = kv_table("Polarization")
    tp.add_row("pol_state", rep.pol_state_repr)
    tp.add_row("pol_transform", rep.pol_transform_repr)
    console.print(tp)

    ts = kv_table("Sampled at solver-time")
    ts.add_row("time_unit_s", f"{rep.time_unit_s:.6g} s")
    ts.add_row("t_eval_solver", f"{rep.t_eval_solver:.6g}")
    ts.add_row("t_eval_phys_s", f"{rep.t_eval_phys_s:.6g} s")
    ts.add_row("E_env_V_m(t_eval)", f"{rep.E_env_eval_V_m:.6g}")

    if (
        rep.E_env_peak_V_m is not None
        and rep.t_peak_solver is not None
        and rep.t_peak_phys_s is not None
    ):
        ts.add_row("t_peak_solver", f"{rep.t_peak_solver:.6g}")
        ts.add_row("t_peak_phys_s", f"{rep.t_peak_phys_s:.6g} s")
        ts.add_row("E_env_peak_V_m", f"{rep.E_env_peak_V_m:.6g}")
    if rep.omega_L_eval_rad_s is None:
        ts.add_row("omega_L_phys", "None")
    else:
        ts.add_row("omega_L_solver", f"{float(rep.omega_L_eval_solver):.6g}")
        ts.add_row(
            "omega_L_phys",
            f"{
                float(rep.omega_L_eval_rad_s):.6g} rad/s",
        )
        ts.add_row(
            "lambda_inferred",
            (
                "None"
                if rep.lambda_inferred_nm is None
                else f"{
                    rep.lambda_inferred_nm:.6g} nm"
            ),
        )
    console.print(ts)

    if (
        show_ascii_plot
        and rep.t_solver is not None
        and rep.E_env_curve_V_m is not None
    ):
        # Keep the ASCII plot, it is surprisingly useful in terminals even with rich tables.
        from .plain import _ascii_plot  # local helper

        console.print("")
        console.print("ASCII pulse (E_env_V_m)")
        for line in _ascii_plot(rep.t_solver, rep.E_env_curve_V_m):
            console.print(line)

    return console.export_text()
