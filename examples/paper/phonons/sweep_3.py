from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
        phi_x1=0.98,
        phi_x2=1.02,
        phi_xx=1.5,
    )

    polaron_la = PolaronLAParams(
        spectral_density=SpectralDensityKind.SUPER_OHMIC_GAUSSIAN,
        enable_polaron_renorm=True,
        enable_polaron_scattering=False,
        alpha=Q(1.0, "ps**2"),
        omega_c=Q(1.0e12, "rad/s"),
        enable_exciton_relaxation=True,
        enable_eid=False,
    )

    phonons = PhononParams(
        kind=PhononModelKind.POLARON_LA,
        temperature=Q(float(T_K), "K"),
        couplings=couplings,
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
    scheme_kwargs: Optional[dict[str, Any]] = None,
    audit: bool = False,
) -> tuple[float, float, float]:
    qd = make_qd_phonons(T_K=float(T_K))

    units, tlist, time_unit_s = _make_units_and_tlist(cfg)
    rho0 = _make_rho0(qd, units)

    factory = get_scheme_factory(scheme)
    specs, _payloads = factory(
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

    m = QDDiagnostics().compute(qd, res, units=units)

    p11 = float(getattr(m.two_photon, "p11"))
    bell = float(getattr(m, "bell_fidelity_phi_plus"))
    lneg = float(getattr(m, "log_negativity_pol"))
    return p11, bell, lneg


def sweep_T_one_metric(
    *,
    cfg: RunCfg,
    scheme: SchemeKind,
    temps_K: np.ndarray,
    amp_scale: float,
    detuning_offset_rad_s: float = 0.0,
    scheme_kwargs: Optional[dict[str, Any]] = None,
    audit: bool = False,
    metric: str = "p11",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (temps_K, y_metric) for a single fixed amp_scale and pulse params.

    metric in {"p11", "bell", "lneg"}.
    """
    temps = np.asarray(temps_K, dtype=float).reshape(-1)
    y = np.full((temps.size,), np.nan, dtype=float)

    for i, T in enumerate(tqdm(temps, desc="Temperature")):
        try:
            p11, bell, lneg = run_one(
                T_K=float(T),
                cfg=cfg,
                scheme=scheme,
                amp_scale=float(amp_scale),
                detuning_offset_rad_s=float(detuning_offset_rad_s),
                scheme_kwargs=scheme_kwargs,
                audit=audit,
            )
        except Exception:
            continue

        if metric == "p11":
            y[i] = float(p11)
        elif metric == "bell":
            y[i] = float(bell)
        elif metric == "lneg":
            y[i] = float(lneg)
        else:
            raise ValueError("Unknown metric: %s" % str(metric))

    return temps, y


def main() -> None:
    cfg = RunCfg(
        t_end_ns=1.0,
        n_points=2001,
        t0_ns=0.2,
        sigma_ns=0.05,
        compensate_polaron=False,
        label_prefix="qd",
    )

    scheme = SchemeKind.TPE
    amp_scale = 2.0
    detuning_offset_rad_s = 0.0

    temps_K = np.linspace(0.0, 50.0, 11)

    metric = "p11"  # "p11" or "bell" or "lneg"
    temps, y = sweep_T_one_metric(
        cfg=cfg,
        scheme=scheme,
        temps_K=temps_K,
        amp_scale=float(amp_scale),
        detuning_offset_rad_s=float(detuning_offset_rad_s),
        scheme_kwargs={},
        audit=False,
        metric=metric,
    )

    outdir = Path("sweeps") / "one_line_vs_T"
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / "temps_K.npy", temps)
    np.save(outdir / ("%s_vs_T.npy" % metric), y)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(temps, y, marker="o", linewidth=1.6)
    ax.set_xlabel("Temperature T (K)")
    if metric == "p11":
        ax.set_ylabel("Two-photon yield p11")
        ax.set_title("p11(T) at fixed amp_scale=%.3g" % float(amp_scale))
    elif metric == "bell":
        ax.set_ylabel("Bell fidelity (phi+)")
        ax.set_title(
            "Bell fidelity(T) at fixed amp_scale=%.3g" % float(amp_scale)
        )
    else:
        ax.set_ylabel("Log negativity")
        ax.set_title(
            "Log negativity(T) at fixed amp_scale=%.3g" % float(amp_scale)
        )

    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / ("%s_vs_T.png" % metric), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
