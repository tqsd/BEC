from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from smef.core.units import Q, hbar, kB
from smef.engine import SimulationEngine, UnitSystem

from bec.core.units import as_rate_1_s
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
from bec.reporting.plotting.labels import ax_label
from bec.reporting.plotting.styles import apply_style, default_style
from bec.scenarios.factories import SchemeKind, get_scheme_factory


# ---------------------------------------------------------------------------
# Global style + solver configuration
# ---------------------------------------------------------------------------

apply_style(default_style())

_SOLVE_OPTIONS: Dict[str, Any] = {
    "qutip_options": {
        "method": "bdf",
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 200000,
        "max_step": 0.01,
        "progress_bar": "tqdm",
    }
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_temperature_sweep_pretty(
    *,
    temps_K: np.ndarray,
    p11: np.ndarray,
    fid: np.ndarray,
    ln_cond: np.ndarray,
    score: np.ndarray,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel, single-column figure:
      top: conditional log-negativity + Bell fidelity (linear)
      bottom: p11 and p11*E_N (log)
    """
    fig_width_in = 3.35
    fig_height_in = 3.10

    x = np.asarray(temps_K, dtype=float).reshape(-1)
    p11 = np.asarray(p11, dtype=float).reshape(-1)
    fid = np.asarray(fid, dtype=float).reshape(-1)
    ln_cond = np.asarray(ln_cond, dtype=float).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace=0.10
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # --- Top panel (linear) ---
    ax_top.plot(
        x,
        ln_cond,
        marker="o",
        linestyle="-",
        linewidth=1.2,
        markersize=3.0,
        label=r"$E_{\mathcal{N}}^{\mathrm{cond}}$",
    )
    ax_top.plot(
        x,
        fid,
        marker="o",
        linestyle="--",
        linewidth=1.2,
        markersize=3.0,
        label=r"$F_{\Phi^+}$",
    )
    ax_top.set_ylabel(
        ax_label("", r"E_{\mathcal{N}}^{\mathrm{cond}},\,F_{\Phi^+}", r"1")
    )

    y_min = float(np.nanmin([np.nanmin(ln_cond), np.nanmin(fid)]))
    y_max = float(np.nanmax([np.nanmax(ln_cond), np.nanmax(fid)]))
    pad = 0.05 * (y_max - y_min + 1e-12)
    ax_top.set_ylim(y_min - pad, y_max + pad)

    ax_top.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_top.tick_params(axis="both", which="major", labelsize=7)
    ax_top.tick_params(labelbottom=False)

    ax_top.legend(
        loc="upper right",
        fontsize=7,
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.3,
    )

    if title:
        ax_top.set_title(title, fontsize=8)

    # --- Bottom panel (log) ---
    ax_bot.plot(
        x,
        p11,
        marker="o",
        linestyle="-",
        linewidth=1.2,
        markersize=3.0,
        label=r"$p_{11}$",
    )
    ax_bot.plot(
        x,
        score,
        marker="o",
        linestyle="--",
        linewidth=1.2,
        markersize=3.0,
        label=r"$p_{11}E_{\mathcal{N}}^{\mathrm{cond}}$",
    )

    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(ax_label("Temperature", "T", r"\kelvin"))
    ax_bot.set_ylabel(
        ax_label("", r"p_{11},\,p_{11}E_{\mathcal{N}}^{\mathrm{cond}}", r"1")
    )

    ax_bot.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_bot.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35)
    ax_bot.tick_params(axis="both", which="major", labelsize=7)

    ax_bot.legend(
        loc="upper right",
        fontsize=7,
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.3,
    )

    ax_bot.margins(x=0.02)
    fig.tight_layout(pad=0.2)
    return fig


def save_fig_pdf(
    fig: plt.Figure, path: Path, *, transparent: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(path),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=bool(transparent),
        metadata={"Creator": "examples.temperature_sweep"},
    )


def write_csv(
    path: Path,
    *,
    temps_K: np.ndarray,
    p11: np.ndarray,
    fid: np.ndarray,
    score: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "T_K,p11,fid_post,score_p11_times_fid\n"
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for T, p, fi, sc in zip(temps_K, p11, fid, score):
            f.write(
                "%.9g,%.18g,%.18g,%.18g\n"
                % (float(T), float(p), float(fi), float(sc))
            )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunCfg:
    t_end_ns: float = 1.0
    n_points: int = 2001
    t0_ns: float = 0.2
    sigma_ns: float = 0.05
    compensate_polaron: bool = False
    label_prefix: str = "qd"


@dataclass(frozen=True)
class TempSweepCfg:
    t_min_K: float = 1.0
    t_max_K: float = 50.0
    n_temps: int = 26
    amp_scale: float = 1.0
    detuning_offset_rad_s: float = 0.0


# ---------------------------------------------------------------------------
# Phonon rate model (simple, explicit)
# ---------------------------------------------------------------------------


def bose_occupation(delta_omega: Any, T: Any) -> float:
    """
    Bose-Einstein occupation n(omega, T) = 1/(exp(hbar*omega/(kB*T)) - 1).

    delta_omega: angular frequency quantity [rad/s]
    T: temperature (float or quantity compatible with kB)
    """
    x = (hbar * delta_omega / (kB * T)).to_base_units().magnitude
    if x > 700:
        return 0.0
    return 1.0 / (math.exp(float(x)) - 1.0)


def J_superohmic_gaussian(omega: Any, *, alpha: Any, omega_c: Any) -> float:
    """
    Super-ohmic Gaussian spectral density:
      J(omega) = alpha * omega^3 * exp(-(omega/omega_c)^2)
    Returns a float with units 1/s (given alpha in s^2 and omega in 1/s).
    """
    w = omega.to("rad/s").magnitude
    wc = omega_c.to("rad/s").magnitude
    a = alpha.to("s**2").magnitude
    return a * (w**3) * math.exp(-((w / wc) ** 2))


def exciton_relax_rates(
    *,
    delta_omega: Any,  # [rad/s]
    T: Any,  # [K] or float compatible with kB
    phi_x1: float,
    phi_x2: float,
    alpha: Any,  # [s^2]
    omega_c: Any,  # [rad/s]
) -> Tuple[float, float]:
    """
    Simple detailed-balance pair:
      gamma_down = Gamma * (n + 1)
      gamma_up   = Gamma * n
    with Gamma = 2*pi*(phi_x1 - phi_x2)^2 * J(delta_omega).
    """
    s2 = (float(phi_x1) - float(phi_x2)) ** 2
    J = J_superohmic_gaussian(delta_omega, alpha=alpha, omega_c=omega_c)
    Gamma = (2.0 * math.pi) * s2 * J
    n = bose_occupation(delta_omega, T)
    gamma_down = Gamma * (n + 1.0)
    gamma_up = Gamma * n
    return gamma_down, gamma_up


# ---------------------------------------------------------------------------
# QD factory (energy structure + cavity + phonons)
# ---------------------------------------------------------------------------


def make_qd_phonons(T_K: float = 4.0) -> QuantumDot:
    # --- Energy + dipoles + cavity ---
    exciton = Q(1.300, "eV")
    binding = Q(3.0e-3, "eV")
    fss = Q(5.0e-6, "eV")

    energy = EnergyStructure.from_params(
        exciton=exciton, binding=binding, fss=fss
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

    # --- Phonon couplings + rate knobs ---
    couplings = PhononCouplings(
        phi_g=0.0,
        phi_x1=0.7,
        phi_x2=1.3,
        phi_xx=2.0,
    )

    alpha = Q(1, "ps**2")
    omega_c = Q(1.0e12, "rad/s")

    delta_omega = (fss.to("J") / hbar).to("1/s")
    g_down, g_up = exciton_relax_rates(
        delta_omega=delta_omega,
        T=float(T_K),
        phi_x1=couplings.phi_x1,
        phi_x2=couplings.phi_x2,
        alpha=alpha,
        omega_c=omega_c,
    )

    # Keep your existing wiring: rates are stored in phenomenological params
    phenom = PhenomenologicalPhononParams(
        gamma_relax_x1_x2=as_rate_1_s(g_down),
        gamma_relax_x2_x1=as_rate_1_s(g_up),
    )
    print(phenom)

    polaron_la = PolaronLAParams(
        spectral_density=SpectralDensityKind.SUPER_OHMIC_GAUSSIAN,
        enable_polaron_renorm=True,
        alpha=alpha,
        omega_c=omega_c,
        enable_exciton_relaxation=True,
        enable_eid=False,
        enable_polaron_scattering=False,
    )

    phonons = PhononParams(
        kind=PhononModelKind.POLARON_LA,
        temperature=Q(float(T_K), "K"),
        couplings=couplings,
        polaron_la=polaron_la,
        phenomenological=phenom,
    )

    return QuantumDot(
        energy=energy, dipoles=dipoles, cavity=cavity, phonons=phonons
    )


# ---------------------------------------------------------------------------
# Simulation plumbing
# ---------------------------------------------------------------------------


def make_units_and_tlist(cfg: RunCfg) -> Tuple[UnitSystem, np.ndarray, float]:
    time_unit_s = float(Q(1.0, "ns").to("s").magnitude)
    units = UnitSystem(time_unit_s=time_unit_s)
    tlist = np.linspace(0.0, float(cfg.t_end_ns), int(cfg.n_points))
    return units, tlist, time_unit_s


def make_rho0(qd: QuantumDot, units: UnitSystem) -> np.ndarray:
    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    return rho0_qd_vacuum(dims=dims, qd_state=QDState.G)


def run_one(
    *,
    qd: QuantumDot,
    cfg: RunCfg,
    scheme: SchemeKind,
    amp_scale: float,
    detuning_offset_rad_s: float = 0.0,
    scheme_kwargs: Optional[Dict[str, Any]] = None,
    audit: bool = False,
) -> Tuple[Any, List[Any], UnitSystem]:
    units, tlist, time_unit_s = make_units_and_tlist(cfg)
    rho0 = make_rho0(qd, units)

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
    return res, list(payloads), units


def print_phonon_sanity(qd: QuantumDot) -> None:
    ph = qd.phonons
    print("=" * 78)
    print("PHONON SANITY")
    print("=" * 78)

    if ph is None:
        print("qd.phonons is None (NO phonons)")
        return

    print("phonon kind:", getattr(ph, "kind", None))
    try:
        print("temperature (K):", float(ph.temperature.to("K").magnitude))
    except Exception:
        print("temperature:", getattr(ph, "temperature", None))

    try:
        rates = dict(qd.phonon_outputs.rates)
        print(
            "phonon_outputs.rates keys:", sorted([str(k) for k in rates.keys()])
        )

        shown = 0
        for k, v in rates.items():
            if shown >= 8:
                break
            try:
                val = float(v.to("1/s").magnitude)
            except Exception:
                val = v
            print("  rate[%s]: %s" % (str(k), str(val)))
            shown += 1

        if shown == 0:
            print(
                "No phonon rates present (maybe only renorm is active, or rates disabled)."
            )
    except Exception as exc:
        print("Could not read phonon_outputs.rates:", repr(exc))

    print("=" * 78)


def run_temperature_sweep(
    *,
    cfg: RunCfg,
    sweep: TempSweepCfg,
    scheme: SchemeKind = SchemeKind.TPE,
    scheme_kwargs: Optional[Dict[str, Any]] = None,
    audit: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    temps_K = np.linspace(
        float(sweep.t_min_K), float(sweep.t_max_K), int(sweep.n_temps)
    )

    p11 = np.empty_like(temps_K, dtype=float)
    fid = np.empty_like(temps_K, dtype=float)
    lns = np.empty_like(temps_K, dtype=float)
    score = np.empty_like(temps_K, dtype=float)

    diag = QDDiagnostics()

    for i, T in enumerate(temps_K):
        qd = make_qd_phonons(T_K=float(T))

        res, _, units = run_one(
            qd=qd,
            cfg=cfg,
            scheme=scheme,
            amp_scale=float(sweep.amp_scale),
            detuning_offset_rad_s=float(sweep.detuning_offset_rad_s),
            scheme_kwargs=scheme_kwargs or {},
            audit=bool(audit),
        )

        m = diag.compute(qd, res, units=units)

        p = float(m.two_photon.p11)
        f = float(m.bell_fidelity_phi_plus)
        ln = float(m.log_negativity_pol)
        purity = float(m.purity_photons_all)
        s = p * ln

        p11[i] = p
        fid[i] = f
        lns[i] = ln
        score[i] = s

        if verbose:
            print(
                "T=%.3f K  p11=%.6g  LN=%.6g  p11*F=%.6g | purity=%.6g"
                % (float(T), p, ln, s, purity)
            )

    return temps_K, p11, fid, lns, score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = RunCfg(
        t_end_ns=1.0,
        n_points=2001,
        t0_ns=0.2,
        sigma_ns=0.05,
        compensate_polaron=False,
        label_prefix="qd",
    )

    sweep = TempSweepCfg(
        t_min_K=1.0,
        t_max_K=50.0,
        n_temps=26,
        amp_scale=1.0,
        detuning_offset_rad_s=0.0,
    )

    # Optional: quick wiring sanity on one representative QD
    qd0 = make_qd_phonons(T_K=float(sweep.t_min_K))
    print_phonon_sanity(qd0)

    temps_K, p11, fid, lns, score = run_temperature_sweep(
        cfg=cfg,
        sweep=sweep,
        scheme=SchemeKind.TPE,
        scheme_kwargs={},
        audit=False,
        verbose=True,
    )

    fig = plot_temperature_sweep_pretty(
        temps_K=temps_K,
        p11=p11,
        fid=fid,
        ln_cond=lns,
        score=score,
        title=None,
    )

    out_dir = Path("out_1")
    save_fig_pdf(fig, out_dir / "temp_sweep_score.pdf")
    write_csv(
        out_dir / "temp_sweep_score.csv",
        temps_K=temps_K,
        p11=p11,
        fid=fid,
        score=score,
    )

    fig.savefig(
        str(out_dir / "temp_sweep_score.png"), dpi=200, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
