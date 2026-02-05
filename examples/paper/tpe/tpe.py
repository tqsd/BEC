from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from smef.core.units import Q
from smef.engine import SimulationEngine, UnitSystem

from bec.metrics.metrics import QDDiagnostics
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.spec.cavity_params import CavityParams
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.reporting.plotting.api import plot_run, plot_runs
from bec.reporting.plotting.grid import PlotConfig
from bec.reporting.plotting.styles import PlotStyle, default_style
from bec.scenarios.factories import SchemeKind, get_scheme_factory

_SOLVE_OPTIONS = {
    "qutip_options": {
        "method": "bdf",
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 200000,
        "max_step": 0.01,
        "progress_bar": "tqdm",
    }
}


@dataclass(frozen=True)
class RunCfg:
    t_end_ns: float = 1.0
    n_points: int = 2001
    t0_ns: float = 0.5
    sigma_ns: float = 0.03
    compensate_polaron: bool = False
    label_prefix: str = "qd"


@dataclass(frozen=True)
class SchemeRun:
    label: str
    res: Any
    payloads: list[Any]


def set_pdf_output_defaults() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def latex_row_from_metrics(
    *,
    label: str,
    m,
    precision: int = 3,
    precision_coh: int = 4,
) -> str:
    """
    Return a single LaTeX table row with \\qty{...}{} wrappers.

    Column order matches:
      Scenario
      N_early
      N_late
      E_N
      E_N_cond
      P
      Lambda
      |rho_pm,mp|
      phase [rad]
    """

    def fmt_num(x: float, p: int) -> str:
        if not np.isfinite(x):
            return r"\text{n/a}"
        return f"{x:.{p}f}"

    def qty(x: float, p: int) -> str:
        s = fmt_num(x, p)
        if s == r"\text{n/a}":
            return s
        return r"\qty{" + s + r"}{}"

    # Brightness
    N_early = float(m.counts.n_xx_total)
    N_late = float(m.counts.n_gx_total)

    # Entanglement
    E_N = float(m.log_negativity_uncond)
    E_N_cond = float(m.log_negativity_pol)

    # Purity
    P = float(m.purity_photons_all)

    # Indistinguishability / overlap
    Lambda = float(m.meta.get("overlap_abs_avg", float("nan")))

    # Coherence + phase
    bc = m.meta.get("bell_component", {}) or {}
    coh = bc.get("coherence_cross", {}) or {}
    coh_abs = float(coh.get("abs", float("nan")))
    phase = float(coh.get("phase_rad", float("nan")))

    # Phase in your example uses 1 decimal; keep it separate if you want
    phase_str = fmt_num(phase, 1)
    phase_cell = (
        r"\qty{" + phase_str + r"}{}"
        if phase_str != r"\text{n/a}"
        else phase_str
    )

    return (
        f"{label}"
        f"     &{qty(N_early, precision)}"
        f"&{qty(N_late, precision)}"
        f"&{qty(E_N, precision)}"
        f"&{qty(E_N_cond, precision)}"
        f"&{qty(P, precision)}"
        f"&{qty(Lambda, precision)}"
        f"&{qty(coh_abs, precision_coh)}"
        f"&{phase_cell}\\\\"
    )


def save_fig_pdf(fig, path: Path, *, transparent: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(path),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=bool(transparent),
        metadata={"Creator": "bec.reporting.plotting"},
    )


def make_qd() -> QuantumDot:
    exciton = Q(1.300, "eV")
    binding = Q(3.0e-3, "eV")
    fss = Q(5.0e-6, "eV")

    energy = EnergyStructure.from_params(
        exciton=exciton, binding=binding, fss=fss
    )

    dipoles = DipoleParams.biexciton_cascade_from_fss(
        mu_default_Cm=Q(3.33564e-29, "C*m"),
        fss=fss,
    )

    cavity = CavityParams.from_values(
        Q=5.0e4,
        Veff_um3=0.5,
        lambda_nm=930.0,
        n=3.4,
    )

    return QuantumDot(
        energy=energy, dipoles=dipoles, cavity=cavity, phonons=None
    )


def make_units_tlist_time_unit(
    cfg: RunCfg,
) -> tuple[UnitSystem, np.ndarray, float]:
    time_unit_s = float(Q(1.0, "ns").to("s").magnitude)
    units = UnitSystem(time_unit_s=time_unit_s)
    tlist = np.linspace(0.0, float(cfg.t_end_ns), int(cfg.n_points))
    return units, tlist, time_unit_s


def make_rho0(qd: QuantumDot, units: UnitSystem) -> np.ndarray:
    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    return rho0_qd_vacuum(dims=dims, qd_state=QDState.G)


def run_scheme(
    *,
    qd: QuantumDot,
    cfg: RunCfg,
    scheme: SchemeKind,
    tlist: np.ndarray,
    time_unit_s: float,
    amp_scale: float,
    detuning_offset_rad_s: float,
    scheme_kwargs: dict[str, Any] | None,
    audit: bool,
) -> tuple[Any, list[Any]]:
    factory = get_scheme_factory(scheme)
    specs, payloads = factory(
        qd,
        cfg=cfg,
        amp_scale=float(amp_scale),
        detuning_offset_rad_s=float(detuning_offset_rad_s),
        label=f"{cfg.label_prefix}_{scheme.name.lower()}",
        **(scheme_kwargs or {}),
    )

    units = UnitSystem(time_unit_s=float(time_unit_s))
    rho0 = make_rho0(qd, units)

    engine = SimulationEngine(audit=bool(audit))
    res = engine.run(
        qd,
        tlist=tlist,
        time_unit_s=float(time_unit_s),
        rho0=rho0,
        drives=specs,
        solve_options=_SOLVE_OPTIONS,
    )
    diag = QDDiagnostics()
    m = diag.compute(qd, res, units=units)
    print(m.to_text(precision=6))
    return res, list(payloads)


def try_run(
    *,
    qd: QuantumDot,
    cfg: RunCfg,
    scheme: SchemeKind,
    label: str,
    tlist: np.ndarray,
    time_unit_s: float,
    amp_scale: float,
    detuning_offset_rad_s: float,
    scheme_kwargs: dict[str, Any] | None,
    audit: bool,
) -> SchemeRun | None:
    try:
        res, payloads = run_scheme(
            qd=qd,
            cfg=cfg,
            scheme=scheme,
            tlist=tlist,
            time_unit_s=time_unit_s,
            amp_scale=amp_scale,
            detuning_offset_rad_s=detuning_offset_rad_s,
            scheme_kwargs=scheme_kwargs,
            audit=audit,
        )
        return SchemeRun(label=label, res=res, payloads=payloads)
    except NotImplementedError as exc:
        print(f"{label} not implemented, skipping: {exc!r}")
        return None
    except Exception as exc:
        traceback.print_exc()
        print(f"{label} failed: {exc!r}")
        return None


def make_a4_style(*, two_column: bool, height_in: float) -> PlotStyle:
    st = default_style()
    return (
        st.a4_two_column(height_in=height_in)
        if two_column
        else st.a4_single_column(height_in=height_in)
    )


def plot_and_save_individual(
    *,
    runs: list[SchemeRun],
    units: UnitSystem,
    qd: QuantumDot,
    out_dir: Path,
    style_single: PlotStyle,
) -> None:
    for r in runs:
        fig = plot_run(
            r.res,
            units=units,
            drives=r.payloads,
            qd=qd,
            cfg=PlotConfig(title=f"TPE case: {r.label}", ncols=1),
            style=style_single,
        )
        save_fig_pdf(fig, out_dir / f"{r.label}.pdf")
        try:
            fig.savefig(
                str(out_dir / f"{r.label}.png"), dpi=200, bbox_inches="tight"
            )
        except Exception:
            pass


def plot_and_save_grid(
    *,
    runs: list[SchemeRun],
    units: UnitSystem,
    qds: list[QuantumDot],
    out_dir: Path,
    style_grid: PlotStyle,
    cfg_grid: PlotConfig,
) -> None:
    results = [r.res for r in runs]
    drives_list = [r.payloads for r in runs]

    figs = plot_runs(
        results,
        units=units,
        drives_list=drives_list,
        qds=qds,
        windows_s=None,
        cfg=cfg_grid,
        style=style_grid,
    )

    for i, fig in enumerate(figs):
        save_fig_pdf(fig, out_dir / f"grid_{i}.pdf")
        try:
            fig.savefig(
                str(out_dir / f"grid_{i}.png"), dpi=200, bbox_inches="tight"
            )
        except Exception:
            pass


def diagnostics(
    *, runs: list[SchemeRun], qd: QuantumDot, units: UnitSystem
) -> None:
    diag = QDDiagnostics()
    for r in runs:
        try:
            m = diag.compute(qd, r.res, units=units)
            print("Diagnostics:", r.label)
            print(m.to_text())
        except Exception as exc:
            print("Diagnostics failed for", r.label + ":", repr(exc))


def main() -> None:
    out_dir = Path("out_tpe_three_cases")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_pdf_output_defaults()

    cfg = RunCfg(
        t_end_ns=1.0,
        n_points=2001,
        t0_ns=0.2,
        sigma_ns=0.05,
        compensate_polaron=False,
        label_prefix="qd",
    )

    qd = make_qd()
    units, tlist, time_unit_s = make_units_tlist_time_unit(cfg)

    # --- Baseline choice ---
    # Pick the "reference" resonant TPE amplitude scale you like.
    # This is the one you consider to be the nominal pi-pulse (or whatever your factory maps to).
    amp_ref = 1.0

    # --- Case 1: resonant TPE ---
    run_resonant = try_run(
        qd=qd,
        cfg=cfg,
        scheme=SchemeKind.TPE,
        label="tpe_resonant",
        tlist=tlist,
        time_unit_s=time_unit_s,
        amp_scale=amp_ref,
        detuning_offset_rad_s=0.0,
        scheme_kwargs={},  # keep defaults
        audit=True,
    )

    # --- Case 2: resonant 5pi pulse ---
    # Implemented as 5x pulse area by scaling amplitude 5x relative to the reference.
    run_5pi = try_run(
        qd=qd,
        cfg=cfg,
        scheme=SchemeKind.TPE,
        label="tpe_resonant_5pi",
        tlist=tlist,
        time_unit_s=time_unit_s,
        amp_scale=5.0 * amp_ref,
        detuning_offset_rad_s=0.0,
        scheme_kwargs={},
        audit=True,
    )

    # --- Case 3: detuned TPE ---
    # Here we detune the two-photon drive by an offset in rad/s.
    # Example: 3 GHz detuning (in cycles/s) -> 2*pi*3e9 rad/s.
    detune_2ph_GHz = 3.0
    detuning_offset_rad_s = float(2.0 * np.pi * detune_2ph_GHz * 1e9)

    run_detuned = try_run(
        qd=qd,
        cfg=cfg,
        scheme=SchemeKind.TPE,
        label=f"tpe_detuned_{detune_2ph_GHz:g}GHz",
        tlist=tlist,
        time_unit_s=time_unit_s,
        amp_scale=amp_ref,
        detuning_offset_rad_s=detuning_offset_rad_s,
        scheme_kwargs={},
        audit=True,
    )

    runs: list[SchemeRun] = [
        r for r in (run_resonant, run_5pi, run_detuned) if r is not None
    ]
    if not runs:
        raise RuntimeError("No runs produced.")

    style_grid = make_a4_style(two_column=True, height_in=5.0)
    style_single = make_a4_style(two_column=False, height_in=5.5)

    cfg_grid = PlotConfig(
        title="",
        ncols=3,
        column_titles=[
            r"TPE resonant $\pi$",
            r"TPE resonant $5\pi$",
            f"TPE detuned ({detune_2ph_GHz:g} GHz)",
        ],
    )

    plot_and_save_individual(
        runs=runs,
        units=units,
        qd=qd,
        out_dir=out_dir,
        style_single=style_single,
    )
    plot_and_save_grid(
        runs=runs,
        units=units,
        qds=[qd for _ in runs],
        out_dir=out_dir,
        style_grid=style_grid,
        cfg_grid=cfg_grid,
    )
    diagnostics(runs=runs, qd=qd, units=units)

    print("\n--- LaTeX table rows (paste directly) ---\n")

    for r, scen_label in zip(
        runs,
        [
            r"$\pi$-pulse",
            r"$5\pi$-pulse",
            r"det.\ $\pi$-pulse",
        ],
    ):
        m = QDDiagnostics().compute(qd, r.res, units=units)
        print(latex_row_from_metrics(label=scen_label, m=m))


if __name__ == "__main__":
    main()
