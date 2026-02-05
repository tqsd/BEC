from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
class RunResult:
    p11: float
    bell: float
    lneg: float


def make_qd_phonons(T_K: float) -> QuantumDot:
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
        alpha=Q(1.0, "ps**2"),
        omega_c=Q(1.0e12, "rad/s"),
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
    detuning_offset_rad_s: float,
    scheme_kwargs: dict[str, Any] | None,
    audit: bool,
    diag: QDDiagnostics,
) -> RunResult:
    qd = make_qd_phonons(T_K=float(T_K))

    units, tlist, time_unit_s = _make_units_and_tlist(cfg)
    rho0 = _make_rho0(qd, units)

    factory = get_scheme_factory(scheme)
    specs, _payloads = factory(
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

    m = diag.compute(qd, res, units=units)

    p11 = float(getattr(m.two_photon, "p11"))
    bell = float(getattr(m, "bell_fidelity_phi_plus"))
    lneg = float(getattr(m, "log_negativity_pol"))
    return RunResult(p11=p11, bell=bell, lneg=lneg)


def find_min_amp_for_yield(
    *,
    T_K: float,
    cfg: RunCfg,
    scheme: SchemeKind,
    amp_grid: np.ndarray,
    p11_target: float,
    detuning_offset_rad_s: float,
    scheme_kwargs: dict[str, Any] | None,
    audit: bool,
    diag: QDDiagnostics,
) -> tuple[float, float]:
    """
    Returns (amp_required, p11_at_amp_required).

    If target is not reached on the grid, returns (nan, max_p11_on_grid).
    """
    best_p11 = np.nan
    amp_required = np.nan
    p11_required = np.nan

    for a in amp_grid:
        try:
            rr = run_one(
                T_K=float(T_K),
                cfg=cfg,
                scheme=scheme,
                amp_scale=float(a),
                detuning_offset_rad_s=float(detuning_offset_rad_s),
                scheme_kwargs=scheme_kwargs,
                audit=audit,
                diag=diag,
            )
        except Exception:
            continue

        p11 = float(rr.p11)
        if not np.isfinite(p11):
            continue

        if not np.isfinite(best_p11) or p11 > best_p11:
            best_p11 = p11

        if p11 >= float(p11_target):
            amp_required = float(a)
            p11_required = p11
            break

    if not np.isfinite(amp_required):
        return np.nan, float(best_p11) if np.isfinite(best_p11) else np.nan

    return amp_required, p11_required


def main() -> None:
    # --- User knobs ---
    scheme = SchemeKind.TPE
    detuning_offset_rad_s = 0.0
    scheme_kwargs: dict[str, Any] = {}

    cfg = RunCfg(
        t_end_ns=1.0,
        n_points=2001,
        t0_ns=0.2,
        sigma_ns=0.05,
        compensate_polaron=False,
        label_prefix="qd",
    )

    # Temperature sweep and amplitude grid
    temps_K = np.linspace(0.0, 50.0, 11)
    amp_grid = np.linspace(0.2, 2.0, 41)

    # Fixed-yield target
    p11_target = 0.85

    outdir = Path("sweeps") / "required_amp_vs_T"
    outdir.mkdir(parents=True, exist_ok=True)

    diag = QDDiagnostics()

    amp_req = np.full((temps_K.size,), np.nan, dtype=float)
    p11_at_req = np.full((temps_K.size,), np.nan, dtype=float)
    p11_best = np.full((temps_K.size,), np.nan, dtype=float)

    for iT, T in enumerate(tqdm(temps_K, desc="Temperature")):
        a, p11_req = find_min_amp_for_yield(
            T_K=float(T),
            cfg=cfg,
            scheme=scheme,
            amp_grid=amp_grid,
            p11_target=float(p11_target),
            detuning_offset_rad_s=float(detuning_offset_rad_s),
            scheme_kwargs=scheme_kwargs,
            audit=False,
            diag=diag,
        )
        amp_req[iT] = float(a)
        p11_at_req[iT] = float(p11_req)

        # also store best achieved p11 on grid for diagnosing "unreachable"
        if np.isfinite(a):
            p11_best[iT] = float(p11_req)
        else:
            p11_best[iT] = float(p11_req)

    # Save raw arrays
    np.save(outdir / "temps_K.npy", temps_K)
    np.save(outdir / "amp_grid.npy", amp_grid)
    np.save(outdir / "amp_required.npy", amp_req)
    np.save(outdir / "p11_at_required.npy", p11_at_req)
    np.save(outdir / "p11_best_on_grid.npy", p11_best)

    # Plot: required amp vs temperature
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    ax.plot(temps_K, amp_req, marker="o")
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel("Minimum amp_scale for p11 >= %.3f" % float(p11_target))
    ax.set_title("Required drive amplitude at fixed two-photon yield")

    # Mark temperatures where target not reached
    bad = ~np.isfinite(amp_req)
    if np.any(bad):
        # Put a small annotation with best p11 achieved at those T
        for T, pbest in zip(temps_K[bad], p11_best[bad]):
            if np.isfinite(pbest):
                ax.annotate(
                    "best p11=%.2f" % float(pbest),
                    (
                        float(T),
                        float(
                            np.nanmax(amp_req[np.isfinite(amp_req)])
                            if np.any(np.isfinite(amp_req))
                            else 0.0
                        ),
                    ),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                )

    fig.tight_layout()
    fig.savefig(outdir / "required_amp_vs_T.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
