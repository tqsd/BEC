from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from smef.core.units import Q
from smef.engine import SimulationEngine, UnitSystem

from bec.metrics.metrics import QDDiagnostics
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.spec.cavity_params import CavityParams
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.phonon_params import (
    PhenomenologicalPhononParams,
    PhononCouplings,
    PhononModelKind,
    PhononParams,
    PolaronLAParams,
    SpectralDensityKind,
)
from bec.scenarios.factories import SchemeKind, get_scheme_factory

_SOLVE_OPTIONS = {
    "qutip_options": {
        "method": "bdf",
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 200000,
        "max_step": 0.01,
        "progress_bar": None,
    }
}


@dataclass(frozen=True)
class RunCfg:
    t_end_ns: float = 1.0
    n_points: int = 2001
    t0_ns: float = 0.2
    sigma_ns: float = 0.05
    compensate_polaron: bool = False
    label_prefix: str = "qd"


@dataclass(frozen=True)
class SchemeRun:
    label: str
    qd: QuantumDot
    res: Any
    payloads: list[Any]
    units: UnitSystem
    time_unit_s: float
    tlist: np.ndarray


@dataclass(frozen=True)
class Sweep2D:
    temps_K: np.ndarray
    amp_scales: np.ndarray
    p11: np.ndarray
    bell: np.ndarray
    lneg: np.ndarray


def make_qd_phonons(T_K: float = 4.0) -> QuantumDot:
    exciton = Q(1.300, "eV")
    binding = Q(3.0e-3, "eV")
    fss = Q(5.0e-6, "eV")

    energy = EnergyStructure.from_params(
        exciton=exciton,
        binding=binding,
        fss=fss,
    )

    dipoles = DipoleParams.biexciton_cascade_from_fss(
        mu_default_Cm=Q(10.0 * 3.33564e-30, "C*m"),
        fss=fss,
    )

    cavity = CavityParams.from_values(
        Q=5.0e4,
        Veff_um3=0.5,
        lambda_nm=930.0,
        n=3.4,
    )

    couplings = PhononCouplings(
        phi_g=0.0,
        phi_x1=1.0,
        phi_x2=1.0,
        phi_xx=1.5,
    )

    phenomenological = PhenomenologicalPhononParams(
        gamma_relax_x1_x2=Q(5e9, "1/s"),
        gamma_relax_x2_x1=Q(5e9, "1/s"),
        gamma_phi_x1=Q(0.0, "1/s"),
        gamma_phi_x2=Q(0.0, "1/s"),
        gamma_phi_xx=Q(0.0, "1/s"),
        gamma_phi_eid_scale=1.0,
    )

    polaron_la = PolaronLAParams(
        spectral_density=SpectralDensityKind.SUPER_OHMIC_GAUSSIAN,
        enable_polaron_renorm=True,
        alpha=Q(0.03, "ps**2"),
        omega_c=Q(1.0e13, "rad/s"),
        enable_exciton_relaxation=True,
        enable_eid=True,
    )

    phonons = PhononParams(
        kind=PhononModelKind.POLARON_LA,
        temperature=Q(float(T_K), "K"),
        couplings=couplings,
        phenomenological=phenomenological,
        polaron_la=polaron_la,
    )

    return QuantumDot(
        energy=energy,
        dipoles=dipoles,
        cavity=cavity,
        phonons=phonons,
    )


def _make_units_and_tlist(cfg: RunCfg) -> tuple[UnitSystem, np.ndarray, float]:
    time_unit_s = float(Q(1.0, "ns").to("s").magnitude)
    units = UnitSystem(time_unit_s=time_unit_s)
    tlist = np.linspace(0.0, float(cfg.t_end_ns), int(cfg.n_points))
    return units, tlist, time_unit_s


def _make_rho0(qd: QuantumDot, units: UnitSystem) -> np.ndarray:
    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    return rho0_qd_vacuum(dims=dims, qd_state=QDState.G)


def run_one(
    *,
    T_K: float,
    cfg: RunCfg,
    scheme: SchemeKind,
    amp_scale: float,
    detuning_offset_rad_s: float = 0.0,
    scheme_kwargs: dict[str, Any] | None = None,
    audit: bool = False,
) -> SchemeRun:
    qd = make_qd_phonons(T_K=float(T_K))

    units, tlist, time_unit_s = _make_units_and_tlist(cfg)
    rho0 = _make_rho0(qd, units)

    factory = get_scheme_factory(scheme)
    specs, payloads = factory(
        qd,
        cfg=cfg,
        amp_scale=float(amp_scale),
        detuning_offset_rad_s=float(detuning_offset_rad_s),
        label=f"{cfg.label_prefix}_{scheme.name.lower()}",
        **(scheme_kwargs or {}),
    )

    engine = SimulationEngine(audit=bool(audit))
    res = engine.run(
        qd,
        tlist=tlist,
        time_unit_s=float(time_unit_s),
        rho0=rho0,
        drives=specs,
        solve_options=_SOLVE_OPTIONS,
    )

    label = f"T={float(T_K):.3g}K_A={float(amp_scale):.3g}"
    return SchemeRun(
        label=label,
        qd=qd,
        res=res,
        payloads=list(payloads),
        units=units,
        time_unit_s=float(time_unit_s),
        tlist=tlist,
    )


def sweep_temperature_amp(
    *,
    cfg: RunCfg,
    scheme: SchemeKind,
    temps_K: np.ndarray,
    amp_scales: np.ndarray,
    detuning_offset_rad_s: float = 0.0,
    scheme_kwargs: dict[str, Any] | None = None,
    audit: bool = False,
) -> Sweep2D:
    temps_K = np.asarray(temps_K, dtype=float).reshape(-1)
    amp_scales = np.asarray(amp_scales, dtype=float).reshape(-1)

    nT = int(temps_K.size)
    nA = int(amp_scales.size)

    p11 = np.full((nT, nA), np.nan, dtype=float)
    bell = np.full((nT, nA), np.nan, dtype=float)
    lneg = np.full((nT, nA), np.nan, dtype=float)

    diag = QDDiagnostics()

    for iT, T in enumerate(tqdm(temps_K, desc="Temperature")):
        for iA, a in enumerate(tqdm(amp_scales, desc="Amp scale", leave=False)):
            try:
                run = run_one(
                    T_K=float(T),
                    cfg=cfg,
                    scheme=scheme,
                    amp_scale=float(a),
                    detuning_offset_rad_s=float(detuning_offset_rad_s),
                    scheme_kwargs=scheme_kwargs,
                    audit=audit,
                )
                m = diag.compute(run.qd, run.res, units=run.units)
            except Exception:
                continue

            try:
                p11[iT, iA] = float(m.two_photon.p11)
            except Exception:
                p11[iT, iA] = np.nan

            try:
                bell[iT, iA] = float(m.bell_fidelity_phi_plus)
            except Exception:
                bell[iT, iA] = np.nan

            try:
                lneg[iT, iA] = float(m.log_negativity_pol)
            except Exception:
                lneg[iT, iA] = np.nan

    return Sweep2D(
        temps_K=temps_K,
        amp_scales=amp_scales,
        p11=p11,
        bell=bell,
        lneg=lneg,
    )


def _mask_by_p11(z: np.ndarray, p11: np.ndarray, p11_min: float) -> np.ndarray:
    out = np.array(z, dtype=float, copy=True)
    out[p11 < float(p11_min)] = np.nan
    return out


def _plot_heatmap(
    *,
    temps_K: np.ndarray,
    amp_scales: np.ndarray,
    z_TA: np.ndarray,
    title: str,
    z_label: str,
    overlay_best_amp: bool = False,
    best_amp_source_TA: np.ndarray | None = None,
) -> plt.Figure:
    # z_TA is shape (nT, nA). We display as amp on y-axis, T on x-axis.
    Z = np.asarray(z_TA, dtype=float).T  # now (nA, nT)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[
            float(temps_K[0]),
            float(temps_K[-1]),
            float(amp_scales[0]),
            float(amp_scales[-1]),
        ],
    )
    ax.set_title(title)
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel("amp_scale")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(z_label)

    if overlay_best_amp:
        src = best_amp_source_TA if best_amp_source_TA is not None else z_TA
        # best amp per temperature: argmax over amp axis
        # handle all-nan rows safely
        best = np.full((temps_K.size,), np.nan, dtype=float)
        for iT in range(int(temps_K.size)):
            row = np.asarray(src[iT, :], dtype=float)
            if np.all(~np.isfinite(row)):
                continue
            j = int(np.nanargmax(row))
            best[iT] = float(amp_scales[j])
        ax.plot(temps_K, best, linewidth=1.6)

    fig.tight_layout()
    return fig


def main() -> None:
    cfg = RunCfg(
        t_end_ns=1.0,
        n_points=2001,
        t0_ns=0.2,
        sigma_ns=0.05,
        compensate_polaron=False,
        label_prefix="qd",
    )

    temps_K = np.linspace(0.0, 50.0, 11)
    amp_scales = np.linspace(0.2, 2.0, 11)

    sweep = sweep_temperature_amp(
        cfg=cfg,
        scheme=SchemeKind.TPE,
        temps_K=temps_K,
        amp_scales=amp_scales,
        detuning_offset_rad_s=0.0,
        scheme_kwargs={},
        audit=False,
    )

    fig1 = _plot_heatmap(
        temps_K=sweep.temps_K,
        amp_scales=sweep.amp_scales,
        z_TA=sweep.p11,
        title="Two-photon yield p11(T, amp_scale)",
        z_label="p11",
        overlay_best_amp=True,
        best_amp_source_TA=sweep.p11,
    )

    bell_masked = _mask_by_p11(sweep.bell, sweep.p11, p11_min=1e-3)
    fig2 = _plot_heatmap(
        temps_K=sweep.temps_K,
        amp_scales=sweep.amp_scales,
        z_TA=bell_masked,
        title="Bell fidelity phi_plus(T, amp_scale), masked where p11 < 1e-3",
        z_label="Bell fidelity (phi+)",
        overlay_best_amp=False,
    )

    lneg_masked = _mask_by_p11(sweep.lneg, sweep.p11, p11_min=1e-3)
    fig3 = _plot_heatmap(
        temps_K=sweep.temps_K,
        amp_scales=sweep.amp_scales,
        z_TA=lneg_masked,
        title="Log negativity(T, amp_scale), masked where p11 < 1e-3",
        z_label="Log negativity",
        overlay_best_amp=False,
    )

    outdir = Path("sweeps") / "TPE_polaron_T_amp"
    outdir.mkdir(parents=True, exist_ok=True)

    fig1.savefig(outdir / "p11.png", dpi=300, bbox_inches="tight")
    fig2.savefig(outdir / "bell_fidelity.png", dpi=300, bbox_inches="tight")
    fig3.savefig(outdir / "log_negativity.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
