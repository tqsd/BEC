from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from smef.core.drives.types import DriveSpec
from smef.core.units import Q
from smef.engine import SimulationEngine
from smef.core.units import UnitSystem

from bec.light.classical.factories import gaussian_field_drive
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.phonon_params import (
    PhononParams,
    PhononModelType,
    PolaronPhononParams,
)


def _sample_drive_envelope_ps(drive: object, t_ps: np.ndarray) -> np.ndarray:
    if not hasattr(drive, "E_env_V_m"):
        raise AttributeError("drive must provide E_env_V_m(t_phys_seconds)")
    t_s = t_ps * 1e-12
    out = np.empty(t_s.size, dtype=float)
    for i in range(t_s.size):
        out[i] = float(drive.E_env_V_m(float(t_s[i])))
    return out


def _get_trace(expect: dict, name: str) -> np.ndarray:
    if name not in expect:
        return np.array([], dtype=float)
    return np.asarray(expect[name], dtype=float)


def _plot_compare(
    *,
    t_ps: np.ndarray,
    drive_E: np.ndarray,
    res_no_ph: object,
    res_ph: object,
    title: str,
) -> None:
    exp0 = getattr(res_no_ph, "expect", {}) or {}
    exp1 = getattr(res_ph, "expect", {}) or {}

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 7), constrained_layout=True
    )

    ax0.plot(t_ps, drive_E, label="E_env (V/m)")
    ax0.set_ylabel("E_env (V/m)")
    ax0.set_title(title)
    ax0.legend()

    pop_keys = ["pop_G", "pop_X1", "pop_X2", "pop_XX"]
    any_pop = False
    for k in pop_keys:
        y0 = _get_trace(exp0, k)
        y1 = _get_trace(exp1, k)
        if y0.size == 0 or y1.size == 0:
            continue
        any_pop = True
        ax1.plot(t_ps, y0, label=f"{k} (no phonons)")
        ax1.plot(t_ps, y1, label=f"{k} (with phonons)")
    if any_pop:
        ax1.set_ylabel("population")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(ncol=2, fontsize=9)
    else:
        ax1.text(
            0.5,
            0.5,
            "No population expectations found",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_ylabel("population")

    n_keys = ["n_GX_H", "n_GX_V", "n_XX_H", "n_XX_V"]
    any_n = False
    for k in n_keys:
        y0 = _get_trace(exp0, k)
        y1 = _get_trace(exp1, k)
        if y0.size == 0 or y1.size == 0:
            continue
        any_n = True
        ax2.plot(t_ps, y0, label=f"{k} (no phonons)")
        ax2.plot(t_ps, y1, label=f"{k} (with phonons)")
    if any_n:
        ax2.set_ylabel("mean n")
        ax2.legend(ncol=2, fontsize=9)
    else:
        ax2.text(
            0.5,
            0.5,
            "No photon-number expectations found",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_ylabel("mean n")

    ax2.set_xlabel("t (ps)")
    plt.show()


def main() -> None:
    # Solver convention: 1 solver unit = 1 ps
    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)

    # Time grid in solver units
    tlist = np.linspace(0.0, 400.0, 2001)  # 0..200 ps, dt=0.1 ps
    t_ps = tlist * (time_unit_s * 1e12)

    # Minimal dot parameters
    energy = EnergyStructure(
        X1=Q(1.200, "eV"),
        X2=Q(1.200, "eV"),
        XX=Q(2.600, "eV"),
    )
    dipoles = DipoleParams(mu_default=Q(5e-28, "C*m"))

    # Two dots: without phonons and with polaron phonons
    qd_no_ph = QuantumDot(energy=energy, dipoles=dipoles, phonons=None)

    phonons = PhononParams(
        model=PhononModelType.POLARON,
        temperature=Q(4.0, "K"),
        phi_G=0.0,
        phi_X=1.0,
        phi_XX=2.0,
        polaron=PolaronPhononParams(
            enable_polaron_renorm=True,
            alpha=Q(0.03, "ps**2"),
            omega_c=Q(1.0e12, "rad/s"),
            enable_exciton_relaxation=False,
        ),
    )
    qd_ph = QuantumDot(energy=energy, dipoles=dipoles, phonons=phonons)

    # One drive payload (field-based). Keep values moderate to avoid stiffness blowups.
    drive = gaussian_field_drive(
        t0=Q(30.0, "ps"),
        sigma=Q(4.0, "ps"),
        E0=Q(0.8e5, "V/m"),
        energy=Q(1.300, "eV"),
        delta_omega=Q(0.0, "rad/s"),
        pol_state=None,
        preferred_kind="2ph",
        label="GX_pulse",
    )

    specs = [DriveSpec(payload=drive, drive_id="GX_pulse")]

    engine = SimulationEngine(audit=False)

    # rho0: ground state and vacuum (dims come from compile_bundle)
    units = UnitSystem(time_unit_s=time_unit_s)

    dims0 = qd_no_ph.compile_bundle(units=units).modes.dims()
    rho0_0 = rho0_qd_vacuum(dims=dims0, qd_state=QDState.G)

    dims1 = qd_ph.compile_bundle(units=units).modes.dims()
    rho0_1 = rho0_qd_vacuum(dims=dims1, qd_state=QDState.G)

    # QuTiP options (tune as needed)
    solve_options = {
        "qutip_options": {
            "method": "bdf",
            "atol": 1e-10,
            "rtol": 1e-8,
            "nsteps": 200000,
            "max_step": 0.05,  # solver units
            "progress_bar": "tqdm",
        }
    }

    res_no_ph = engine.run(
        qd_no_ph,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0_0,
        drives=specs,
        solve_options=solve_options,
    )

    res_ph = engine.run(
        qd_ph,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0_1,
        drives=specs,
        solve_options=solve_options,
    )

    # Drive envelope (same for both, plotted once)
    E = _sample_drive_envelope_ps(drive, t_ps)

    _plot_compare(
        t_ps=t_ps,
        drive_E=E,
        res_no_ph=res_no_ph,
        res_ph=res_ph,
        title="QuantumDot: comparison (no phonons) vs (polaron phonons enabled)",
    )


if __name__ == "__main__":
    main()
