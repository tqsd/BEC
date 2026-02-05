from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

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
    omega_solver: np.ndarray  # complex, solver units, B-renormalized


@dataclass(frozen=True)
class Sweep2D:
    temps_K: np.ndarray
    amp_scales: np.ndarray

    # original metrics
    p11: np.ndarray
    bell: np.ndarray
    lneg: np.ndarray

    # new "story" metrics
    # integral (g_down+g_up+g_cd) dt (dimensionless)
    scatt_budget: np.ndarray
    scatt_budget_down: np.ndarray  # integral g_down dt
    scatt_budget_up: np.ndarray  # integral g_up dt
    scatt_budget_cd: np.ndarray  # integral g_cd dt
    scatt_over_drive: np.ndarray  # scatt_budget / int Omega_R^2 dt


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
        alpha=Q(0.3, "ps**2"),
        omega_c=Q(1.0e12, "rad/s"),
        enable_exciton_relaxation=True,
        enable_polaron_scattering=True,
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


def _extract_omega_solver_from_payloads(
    *,
    qd: QuantumDot,
    payloads: list[Any],
    tlist: np.ndarray,
    time_unit_s: float,
    scheme: SchemeKind,
) -> np.ndarray:
    """
    Best-effort: reconstruct the solver Omega(t) on the grid for the *main* driven pair.

    We intentionally do not depend on SMEF internals here. We call the payload if it
    exposes E_env_V_m(t) and reuse qd.derived_view.mu and drive projection the same
    way your DriveStrengthModel does.

    Assumption for TPE: the drive targets TransitionPair.G_XX and uses the forward
    direction for absorption. If you have other schemes, adapt this mapping.
    """
    # If your factory emits exactly one payload for the main drive, take first.
    if not payloads:
        return np.zeros((tlist.size,), dtype=complex)

    payload = payloads[0]
    tlist_f = np.asarray(tlist, dtype=float).reshape(-1)
    s = float(time_unit_s)
    t_phys_s = s * tlist_f

    fn_E = getattr(payload, "E_env_V_m", None)
    if not callable(fn_E):
        return np.zeros((tlist.size,), dtype=complex)

    E_t = np.empty(t_phys_s.size, dtype=float)
    for i in range(t_phys_s.size):
        E_t[i] = float(fn_E(float(t_phys_s[i])))

    # placeholder; overwritten below
    hbar_Js = float(Q(1.0, "J*s").to("J*s").magnitude)
    try:
        from smef.core.units import hbar, magnitude

        hbar_Js = float(magnitude(hbar, "J*s"))
    except Exception:
        # fallback constant
        hbar_Js = 1.054571817e-34

    derived = qd.derived_view

    # Map scheme -> intended pair. For your example: TPE drives G<->XX.
    pair = None
    if scheme is SchemeKind.TPE:
        from bec.quantum_dot.enums import TransitionPair

        pair = TransitionPair.G_XX

    if pair is None:
        return np.zeros((tlist.size,), dtype=complex)

    fwd, _ = derived.t_registry.directed(pair)
    mu_Cm = float(derived.mu_Cm(fwd))

    pol = 1.0 + 0.0j
    pol_vec = None
    fn_pol = getattr(payload, "effective_pol", None)
    if callable(fn_pol):
        out = fn_pol()
        if out is not None:
            pol_vec = np.asarray(out, dtype=complex).reshape(2)
    if pol_vec is not None:
        pol = complex(derived.drive_projection(fwd, pol_vec))

    omega_rad_s = (mu_Cm * E_t) / hbar_Js
    omega_solver = (omega_rad_s * s).astype(complex) * pol

    # Polaron renormalization is already in derived.polaron_B if enabled.
    B = float(derived.polaron_B(fwd))
    omega_solver = omega_solver * (B + 0.0j)

    return np.asarray(omega_solver, dtype=complex).reshape(-1)


def _compute_polaron_scatt_budgets(
    *,
    qd: QuantumDot,
    omega_solver: np.ndarray,
    tlist: np.ndarray,
    time_unit_s: float,
    detuning_rad_s: np.ndarray,
    Nt: int = 4096,
) -> tuple[float, float, float, float, float]:
    """
    Returns:
      (budget_total, budget_down, budget_up, budget_cd, scatt_over_drive)

    budget_* = integral g_*(t) dt_phys in dimensionless units.
    Because dt_phys = time_unit_s * dt_solver and g is in 1/s, this is dimensionless.
    """
    derived = qd.derived_view
    po = getattr(derived, "phonon_outputs", None)
    if po is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    polaron_rates = getattr(po, "polaron_rates", None)
    if polaron_rates is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if not bool(getattr(polaron_rates, "enabled", False)):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    tlist_f = np.asarray(tlist, dtype=float).reshape(-1)
    s = float(time_unit_s)
    t_phys = s * tlist_f

    omega_solver = np.asarray(omega_solver, dtype=complex).reshape(-1)
    detuning_rad_s = np.asarray(detuning_rad_s, dtype=float).reshape(-1)
    if omega_solver.size != tlist_f.size:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    if detuning_rad_s.size != tlist_f.size:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Physical rates in 1/s
    g_down, g_up, g_cd = polaron_rates.gamma_dressed_rates_1_s(
        omega_solver=omega_solver,
        detuning_rad_s=detuning_rad_s,
        time_unit_s=s,
        b_polaron=1.0,  # omega_solver already B-renormalized
        Nt=int(Nt),
    )
    g_down = np.asarray(g_down, dtype=float).reshape(-1)
    g_up = np.asarray(g_up, dtype=float).reshape(-1)
    g_cd = np.asarray(g_cd, dtype=float).reshape(-1)

    # Integrals in dimensionless units: int g(t) dt_phys
    b_down = float(np.trapezoid(g_down, t_phys))
    b_up = float(np.trapezoid(g_up, t_phys))
    b_cd = float(np.trapezoid(g_cd, t_phys))
    b_tot = float(b_down + b_up + b_cd)

    # Normalize by drive "energy": int Omega_R^2 dt_phys
    # Omega_R is the physical Rabi frequency magnitude.
    om_phys = omega_solver / s
    OmR = np.sqrt((om_phys.real * om_phys.real) + (om_phys.imag * om_phys.imag))
    drive_energy = float(np.trapezoid(OmR * OmR, t_phys))
    if drive_energy <= 0.0:
        ratio = np.nan
    else:
        ratio = float(b_tot / drive_energy)

    return b_tot, b_down, b_up, b_cd, ratio


def run_one(
    *,
    T_K: float,
    cfg: RunCfg,
    scheme: SchemeKind,
    amp_scale: float,
    detuning_offset_rad_s: float = 0.0,
    scheme_kwargs: Optional[dict[str, Any]] = None,
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
        label="%s_%s" % (cfg.label_prefix, scheme.name.lower()),
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

    omega_solver = _extract_omega_solver_from_payloads(
        qd=qd,
        payloads=list(payloads),
        tlist=tlist,
        time_unit_s=float(time_unit_s),
        scheme=scheme,
    )

    label = "T=%.3gK_A=%.3g" % (float(T_K), float(amp_scale))
    return SchemeRun(
        label=label,
        qd=qd,
        res=res,
        payloads=list(payloads),
        units=units,
        time_unit_s=float(time_unit_s),
        tlist=np.asarray(tlist, dtype=float),
        omega_solver=omega_solver,
    )


def sweep_temperature_amp(
    *,
    cfg: RunCfg,
    scheme: SchemeKind,
    temps_K: np.ndarray,
    amp_scales: np.ndarray,
    detuning_offset_rad_s: float = 0.0,
    scheme_kwargs: Optional[dict[str, Any]] = None,
    audit: bool = False,
    Nt_rates: int = 4096,
) -> Sweep2D:
    temps_K = np.asarray(temps_K, dtype=float).reshape(-1)
    amp_scales = np.asarray(amp_scales, dtype=float).reshape(-1)

    nT = int(temps_K.size)
    nA = int(amp_scales.size)

    p11 = np.full((nT, nA), np.nan, dtype=float)
    bell = np.full((nT, nA), np.nan, dtype=float)
    lneg = np.full((nT, nA), np.nan, dtype=float)

    scatt_budget = np.full((nT, nA), np.nan, dtype=float)
    scatt_budget_down = np.full((nT, nA), np.nan, dtype=float)
    scatt_budget_up = np.full((nT, nA), np.nan, dtype=float)
    scatt_budget_cd = np.full((nT, nA), np.nan, dtype=float)
    scatt_over_drive = np.full((nT, nA), np.nan, dtype=float)

    diag = QDDiagnostics()

    # For now we treat detuning_offset as a constant detuning array on the grid.
    # If your scheme uses time-dependent detuning, replace here.
    # detuning_rad_s(t) = detuning_offset_rad_s
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

            # Original metrics
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

            # New budgets from polaron rates
            detuning = np.full(
                run.tlist.size, float(detuning_offset_rad_s), dtype=float
            )
            b_tot, b_d, b_u, b_cd, ratio = _compute_polaron_scatt_budgets(
                qd=run.qd,
                omega_solver=run.omega_solver,
                tlist=run.tlist,
                time_unit_s=run.time_unit_s,
                detuning_rad_s=detuning,
                Nt=int(Nt_rates),
            )
            scatt_budget[iT, iA] = float(b_tot)
            scatt_budget_down[iT, iA] = float(b_d)
            scatt_budget_up[iT, iA] = float(b_u)
            scatt_budget_cd[iT, iA] = float(b_cd)
            scatt_over_drive[iT, iA] = float(ratio)

    return Sweep2D(
        temps_K=temps_K,
        amp_scales=amp_scales,
        p11=p11,
        bell=bell,
        lneg=lneg,
        scatt_budget=scatt_budget,
        scatt_budget_down=scatt_budget_down,
        scatt_budget_up=scatt_budget_up,
        scatt_budget_cd=scatt_budget_cd,
        scatt_over_drive=scatt_over_drive,
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
    best_amp_source_TA: Optional[np.ndarray] = None,
) -> plt.Figure:
    # z_TA is shape (nT, nA). We display as amp on y-axis, T on x-axis.
    Z = np.asarray(z_TA, dtype=float).T  # (nA, nT)

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


def _plot_frontier_best_bell_at_yield(
    *,
    temps_K: np.ndarray,
    amp_scales: np.ndarray,
    p11_TA: np.ndarray,
    bell_TA: np.ndarray,
    p11_min: float,
    title: str,
) -> plt.Figure:
    """
    For each temperature, find max Bell fidelity among amp values with p11 >= p11_min.
    This often shows "even if you crank power, best achievable quality drops".
    """
    temps_K = np.asarray(temps_K, dtype=float).reshape(-1)
    best = np.full((temps_K.size,), np.nan, dtype=float)
    best_amp = np.full((temps_K.size,), np.nan, dtype=float)

    for iT in range(int(temps_K.size)):
        p = np.asarray(p11_TA[iT, :], dtype=float)
        b = np.asarray(bell_TA[iT, :], dtype=float)
        ok = np.isfinite(p) & np.isfinite(b) & (p >= float(p11_min))
        if not np.any(ok):
            continue
        j = int(np.nanargmax(np.where(ok, b, np.nan)))
        best[iT] = float(b[j])
        best_amp[iT] = float(amp_scales[j])

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(temps_K, best, marker="o", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel("best Bell fidelity (phi+)")
    ax.grid(True, linewidth=0.4, alpha=0.4)
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

    temps_K = np.linspace(0.0, 50.0, 5)
    amp_scales = np.linspace(0.2, 2.0, 5)

    sweep = sweep_temperature_amp(
        cfg=cfg,
        scheme=SchemeKind.TPE,
        temps_K=temps_K,
        amp_scales=amp_scales,
        detuning_offset_rad_s=0.0,
        scheme_kwargs={},
        audit=False,
        Nt_rates=4096,
    )

    outdir = Path("sweeps") / "TPE_polaron_T_amp"
    outdir.mkdir(parents=True, exist_ok=True)

    # Original plots
    fig1 = _plot_heatmap(
        temps_K=sweep.temps_K,
        amp_scales=sweep.amp_scales,
        z_TA=sweep.p11,
        title="Two-photon yield p11(T, amp_scale)",
        z_label="p11",
        overlay_best_amp=True,
        best_amp_source_TA=sweep.p11,
    )
    fig1.savefig(outdir / "p11.png", dpi=300, bbox_inches="tight")

    bell_masked = _mask_by_p11(sweep.bell, sweep.p11, p11_min=1e-3)
    fig2 = _plot_heatmap(
        temps_K=sweep.temps_K,
        amp_scales=sweep.amp_scales,
        z_TA=bell_masked,
        title="Bell fidelity phi_plus(T, amp_scale), masked where p11 < 1e-3",
        z_label="Bell fidelity (phi+)",
        overlay_best_amp=False,
    )
    fig2.savefig(outdir / "bell_fidelity.png", dpi=300, bbox_inches="tight")

    lneg_masked = _mask_by_p11(sweep.lneg, sweep.p11, p11_min=1e-3)
    fig3 = _plot_heatmap(
        temps_K=sweep.temps_K,
        amp_scales=sweep.amp_scales,
        z_TA=lneg_masked,
        title="Log negativity(T, amp_scale), masked where p11 < 1e-3",
        z_label="Log negativity",
        overlay_best_amp=False,
    )
    fig3.savefig(outdir / "log_negativity.png", dpi=300, bbox_inches="tight")

    # New: phonon scattering budgets
    fig4 = _plot_heatmap(
        temps_K=sweep.temps_K,
        amp_scales=sweep.amp_scales,
        z_TA=sweep.scatt_budget,
        title="Polaron dressed scattering budget int (g_down+g_up+g_cd) dt",
        z_label="dimensionless budget",
        overlay_best_amp=False,
    )
    fig4.savefig(
        outdir / "scatt_budget_total.png", dpi=300, bbox_inches="tight"
    )

    fig5 = _plot_heatmap(
        temps_K=sweep.temps_K,
        amp_scales=sweep.amp_scales,
        z_TA=sweep.scatt_over_drive,
        title="Polaron scattering budget normalized by int Omega_R^2 dt",
        z_label="budget / drive_energy",
        overlay_best_amp=False,
    )
    fig5.savefig(outdir / "scatt_over_drive.png", dpi=300, bbox_inches="tight")

    # New: frontier plot (often the cleanest narrative figure)
    fig6 = _plot_frontier_best_bell_at_yield(
        temps_K=sweep.temps_K,
        amp_scales=sweep.amp_scales,
        p11_TA=sweep.p11,
        bell_TA=sweep.bell,
        p11_min=0.5,
        title="Best achievable Bell fidelity at fixed yield (p11 >= 0.5)",
    )
    fig6.savefig(
        outdir / "best_bell_given_yield.png", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
