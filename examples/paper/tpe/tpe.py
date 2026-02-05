from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

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


# ---------------------------------------------------------------------------
# Solver + output defaults
# ---------------------------------------------------------------------------

_SOLVE_OPTIONS: dict[str, Any] = {
    "qutip_options": {
        "method": "bdf",
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 200000,
        "max_step": 0.01,
        "progress_bar": "tqdm",
    }
}


def set_pdf_output_defaults() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def save_fig_pdf(fig: Any, path: Path, *, transparent: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(path),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=bool(transparent),
        metadata={"Creator": "bec.reporting.plotting"},
    )


def save_fig_png(fig: Any, path: Path, *, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=int(dpi), bbox_inches="tight")


# ---------------------------------------------------------------------------
# Config + run container
# ---------------------------------------------------------------------------


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
    scheme: SchemeKind
    amp_scale: float
    detuning_offset_rad_s: float
    scheme_kwargs: dict[str, Any]
    res: Any
    payloads: list[Any]


# ---------------------------------------------------------------------------
# LaTeX table row helper
# ---------------------------------------------------------------------------


def latex_row_from_metrics(
    *,
    label: str,
    m: Any,
    precision: int = 3,
    precision_coh: int = 4,
) -> str:
    """
    Return a LaTeX table row with \\qty{...}{} wrappers.

    Column order:
      Scenario, N_early, N_late, E_N, E_N_cond, P, Lambda, |rho_pm,mp|, phase [rad]
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

    N_early = float(m.counts.n_xx_total)
    N_late = float(m.counts.n_gx_total)

    E_N = float(m.log_negativity_uncond)
    E_N_cond = float(m.log_negativity_pol)

    P = float(m.purity_photons_all)
    Lambda = float(m.meta.get("overlap_abs_avg", float("nan")))

    bc = m.meta.get("bell_component", {}) or {}
    coh = bc.get("coherence_cross", {}) or {}
    coh_abs = float(coh.get("abs", float("nan")))
    phase = float(coh.get("phase_rad", float("nan")))

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


# ---------------------------------------------------------------------------
# QD + simulation plumbing
# ---------------------------------------------------------------------------


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
    scheme_kwargs: Optional[dict[str, Any]],
    audit: bool,
    verbose_metrics: bool = True,
) -> tuple[Any, list[Any], UnitSystem]:
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

    if verbose_metrics:
        diag = QDDiagnostics()
        m = diag.compute(qd, res, units=units)
        try:
            print(m.to_text(precision=6))
        except Exception:
            print(m)

    return res, list(payloads), units


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
    scheme_kwargs: Optional[dict[str, Any]],
    audit: bool,
) -> Optional[SchemeRun]:
    try:
        res, payloads, _ = run_scheme(
            qd=qd,
            cfg=cfg,
            scheme=scheme,
            tlist=tlist,
            time_unit_s=time_unit_s,
            amp_scale=amp_scale,
            detuning_offset_rad_s=detuning_offset_rad_s,
            scheme_kwargs=scheme_kwargs,
            audit=audit,
            verbose_metrics=True,
        )
        return SchemeRun(
            label=label,
            scheme=scheme,
            amp_scale=float(amp_scale),
            detuning_offset_rad_s=float(detuning_offset_rad_s),
            scheme_kwargs=dict(scheme_kwargs or {}),
            res=res,
            payloads=payloads,
        )
    except NotImplementedError as exc:
        print(f"{label} not implemented, skipping: {exc!r}")
        return None
    except Exception as exc:
        traceback.print_exc()
        print(f"{label} failed: {exc!r}")
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_a4_style(*, two_column: bool, height_in: float) -> PlotStyle:
    st = default_style()
    if two_column:
        return st.a4_two_column(height_in=height_in)
    return st.a4_single_column(height_in=height_in)


def plot_and_save_individual(
    *,
    runs: Sequence[SchemeRun],
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
            save_fig_png(fig, out_dir / f"{r.label}.png")
        except Exception:
            pass


def plot_and_save_grid(
    *,
    runs: Sequence[SchemeRun],
    units: UnitSystem,
    qds: Sequence[QuantumDot],
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
        qds=list(qds),
        windows_s=None,
        cfg=cfg_grid,
        style=style_grid,
    )

    for i, fig in enumerate(figs):
        save_fig_pdf(fig, out_dir / f"grid_{i}.pdf")
        try:
            save_fig_png(fig, out_dir / f"grid_{i}.png")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CSV export for reproducibility (one row per case)
# ---------------------------------------------------------------------------


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def write_repro_csv(
    path: Path,
    *,
    runs: Sequence[SchemeRun],
    qd: QuantumDot,
    units: UnitSystem,
) -> None:
    """
    Write one row per case with the key quantities used in your plots/tables.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    diag = QDDiagnostics()

    cols = [
        "label",
        "scheme",
        "amp_scale",
        "detuning_offset_rad_s",
        "n_xx_total",
        "n_gx_total",
        "log_negativity_uncond",
        "log_negativity_pol",
        "purity_photons_all",
        "overlap_abs_avg",
        "bell_coh_abs",
        "bell_coh_phase_rad",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")

        for r in runs:
            try:
                m = diag.compute(qd, r.res, units=units)

                n_xx = _safe_float(
                    getattr(m.counts, "n_xx_total", float("nan"))
                )
                n_gx = _safe_float(
                    getattr(m.counts, "n_gx_total", float("nan"))
                )
                ln_un = _safe_float(
                    getattr(m, "log_negativity_uncond", float("nan"))
                )
                ln_pol = _safe_float(
                    getattr(m, "log_negativity_pol", float("nan"))
                )
                purity = _safe_float(
                    getattr(m, "purity_photons_all", float("nan"))
                )

                overlap = _safe_float(
                    m.meta.get("overlap_abs_avg", float("nan"))
                )

                bc = m.meta.get("bell_component", {}) or {}
                coh = bc.get("coherence_cross", {}) or {}
                coh_abs = _safe_float(coh.get("abs", float("nan")))
                coh_phase = _safe_float(coh.get("phase_rad", float("nan")))

                row = [
                    r.label,
                    r.scheme.name,
                    f"{r.amp_scale:.18g}",
                    f"{r.detuning_offset_rad_s:.18g}",
                    f"{n_xx:.18g}",
                    f"{n_gx:.18g}",
                    f"{ln_un:.18g}",
                    f"{ln_pol:.18g}",
                    f"{purity:.18g}",
                    f"{overlap:.18g}",
                    f"{coh_abs:.18g}",
                    f"{coh_phase:.18g}",
                ]
                f.write(",".join(row) + "\n")
            except Exception as exc:
                print("CSV export failed for", r.label + ":", repr(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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

    amp_ref = 1.0

    # Case 1: resonant pi-ish (reference)
    run_resonant = try_run(
        qd=qd,
        cfg=cfg,
        scheme=SchemeKind.TPE,
        label="tpe_resonant",
        tlist=tlist,
        time_unit_s=time_unit_s,
        amp_scale=amp_ref,
        detuning_offset_rad_s=0.0,
        scheme_kwargs={},
        audit=True,
    )

    # Case 2: resonant 5pi (amplitude scaled)
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

    # Case 3: detuned pi-ish
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

    runs = [r for r in (run_resonant, run_5pi, run_detuned) if r is not None]
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

    # CSV export for reproducing the summary plot/table
    write_repro_csv(
        out_dir / "tpe_three_cases_repro.csv", runs=runs, qd=qd, units=units
    )

    print("\n--- LaTeX table rows (paste directly) ---\n")
    scen_labels = {
        "tpe_resonant": r"$\pi$-pulse",
        "tpe_resonant_5pi": r"$5\pi$-pulse",
        f"tpe_detuned_{detune_2ph_GHz:g}GHz": r"det.\ $\pi$-pulse",
    }

    diag = QDDiagnostics()
    for r in runs:
        m = diag.compute(qd, r.res, units=units)
        print(
            latex_row_from_metrics(label=scen_labels.get(r.label, r.label), m=m)
        )


if __name__ == "__main__":
    main()
