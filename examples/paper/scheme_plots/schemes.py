from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    payloads: List[Any]


def set_pdf_output_defaults() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


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
) -> Tuple[UnitSystem, np.ndarray, float]:
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
    scheme_kwargs: Optional[Dict[str, Any]],
    audit: bool,
) -> Tuple[Any, List[Any]]:
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
    scheme_kwargs: Optional[Dict[str, Any]],
    audit: bool,
) -> Optional[SchemeRun]:
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
    runs: List[SchemeRun],
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
            cfg=PlotConfig(title=f"QD scheme: {r.label}", ncols=1),
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
    runs: List[SchemeRun],
    units: UnitSystem,
    qds: List[QuantumDot],
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
    *, runs: List[SchemeRun], qd: QuantumDot, units: UnitSystem
) -> None:
    diag = QDDiagnostics()
    for r in runs:
        try:
            m = diag.compute(qd, r.res, units=units)
            print("Diagnostics:", r.label)
            print(m)
        except Exception as exc:
            print("Diagnostics failed for", r.label + ":", repr(exc))


def main() -> None:
    out_dir = Path("out_three_schemes")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_pdf_output_defaults()

    cfg = RunCfg(
        t_end_ns=1.0,
        n_points=2001,
        t0_ns=0.2,
        sigma_ns=0.02,
        compensate_polaron=False,
        label_prefix="qd",
    )

    detuning_offset_rad_s = 0.0

    bichromatic_kwargs = {
        "dt_ns": -0.030,
        "sigma_gx_ns": 0.03,
        "sigma_xx_ns": 0.03,
        "dpe_delta_rad_s": 2.0e11,
    }

    arp_kwargs = {
        "chirp_kind": "tanh",
        "tanh_delta_rad_s": 5.0e10,
        "tanh_tau_ns": 0.02,
        "sigma_ns": 0.05,
    }

    qd = make_qd()
    units, tlist, time_unit_s = make_units_tlist_time_unit(cfg)

    runs: List[SchemeRun] = []
    for r in (
        try_run(
            qd=qd,
            cfg=cfg,
            scheme=SchemeKind.TPE,
            label="tpe",
            tlist=tlist,
            time_unit_s=time_unit_s,
            amp_scale=4.0,
            detuning_offset_rad_s=detuning_offset_rad_s,
            scheme_kwargs={},
            audit=True,
        ),
        try_run(
            qd=qd,
            cfg=cfg,
            scheme=SchemeKind.BICHROMATIC,
            label="bichromatic",
            tlist=tlist,
            time_unit_s=time_unit_s,
            amp_scale=4.0,
            detuning_offset_rad_s=detuning_offset_rad_s,
            scheme_kwargs=bichromatic_kwargs,
            audit=True,
        ),
        try_run(
            qd=qd,
            cfg=cfg,
            scheme=SchemeKind.ARP,
            label="arp",
            tlist=tlist,
            time_unit_s=time_unit_s,
            amp_scale=1.0,
            detuning_offset_rad_s=detuning_offset_rad_s,
            scheme_kwargs=arp_kwargs,
            audit=True,
        ),
    ):
        if r is not None:
            runs.append(r)

    if not runs:
        raise RuntimeError("No runs produced.")

    style_grid = make_a4_style(two_column=True, height_in=6.0)
    style_single = make_a4_style(two_column=False, height_in=5.5)

    cfg_grid = PlotConfig(
        title="",
        ncols=3,
        column_titles=["TPE", "STIRAP", "ARP"],
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


if __name__ == "__main__":
    main()
