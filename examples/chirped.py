from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from smef.core.drives.types import DriveSpec
from smef.engine import SimulationEngine, UnitSystem
from smef.core.units import Q

from bec.light.classical.factories import gaussian_field_drive
from bec.light.classical.field_drive import ClassicalFieldDriveU
from bec.light.classical.carrier import Carrier
from bec.light.classical import carrier_profiles

from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState, TransitionPair
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.dipole_params import DipoleParams


def plot_qd_run_summary(
    res,
    *,
    time_unit_s: float,
    drive: object,
    envelope_label: str = "E_env (V/m)",
) -> None:
    """
    One figure with 3 stacked plots:
      1) pulse envelope (from drive.E_env_V_m(t_phys))
      2) QD populations (pop_G, pop_X1, pop_X2, pop_XX)
      3) photon numbers (n_GX_H, n_GX_V, n_XX_H, n_XX_V)

    Assumes:
      - res.tlist exists (solver units)
      - res.expect is a mapping key -> 1D array aligned with tlist
      - drive.E_env_V_m(t_phys_seconds) exists
    """
    t_solver = np.asarray(getattr(res, "tlist", None), dtype=float)
    if t_solver.ndim != 1 or t_solver.size == 0:
        raise ValueError("res.tlist must be a non-empty 1D array")

    s = float(time_unit_s)
    t_s = t_solver * s
    t_ps = t_s * 1e12

    expect = getattr(res, "expect", None)
    if expect is None:
        raise AttributeError("res has no attribute 'expect'")

    def _trace(name: str) -> np.ndarray:
        if name not in expect:
            return np.array([], dtype=float)
        y = np.asarray(expect[name], dtype=float)
        if y.shape[0] != t_solver.shape[0]:
            raise ValueError(
                f"Trace {name} has length {
                    y.shape[0]} but t has {t_solver.shape[0]}"
            )
        return y

    # ---- top: pulse envelope ----
    if not hasattr(drive, "E_env_V_m"):
        raise AttributeError("drive must provide E_env_V_m(t_phys_seconds)")

    E = np.asarray(
        [float(drive.E_env_V_m(float(ts))) for ts in t_s], dtype=float
    )

    # ---- middle: QD populations ----
    pop_keys = ["pop_G", "pop_X1", "pop_X2", "pop_XX"]
    pops = {k: _trace(k) for k in pop_keys if k in expect}

    # ---- bottom: photon numbers ----
    n_keys = ["n_GX_H", "n_GX_V", "n_XX_H", "n_XX_V"]
    nums = {k: _trace(k) for k in n_keys if k in expect}

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 7), constrained_layout=True
    )

    # Top plot: envelope
    ax0.plot(t_ps, E, label=envelope_label)
    ax0.set_ylabel(envelope_label)
    ax0.set_title("Drive envelope, QD populations, photon numbers")
    ax0.legend()

    # Middle plot: QD populations
    if pops:
        for k, y in pops.items():
            ax1.plot(t_ps, y, label=k)
        ax1.set_ylabel("population")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend()
    else:
        ax1.text(
            0.5,
            0.5,
            "No QD population expectations found",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_ylabel("population")

    # Bottom plot: photon numbers
    if nums:
        for k, y in nums.items():
            ax2.plot(t_ps, y, label=k)
        ax2.set_ylabel("mean n")
        ax2.legend()
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


def plot_drive_omega(
    drive: ClassicalFieldDriveU, *, tlist: np.ndarray, time_unit_s: float
) -> None:
    t_solver = np.asarray(tlist, dtype=float).reshape(-1)
    t_s = t_solver * float(time_unit_s)
    w = np.asarray([drive.omega_L_rad_s(float(ts)) for ts in t_s], dtype=float)

    plt.figure()
    plt.plot(t_s * 1e12, w)
    plt.xlabel("t (ps)")
    plt.ylabel("omega_L(t) (rad/s)")
    plt.title("Instantaneous carrier frequency")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # --- QD spec ---
    energy = EnergyStructure(
        X1=Q(1.201, "eV"),
        X2=Q(1.201, "eV"),
        XX=Q(2.600, "eV"),
    )
    dipoles = DipoleParams(mu_default=Q(1e-27, "C*m"))
    qd = QuantumDot(energy=energy, dipoles=dipoles)

    # --- solver grid ---
    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)
    tlist = np.linspace(0.0, 200.0, 2001)  # solver units

    # --- choose a target for setting omega0 ---
    # For a 2ph drive of G<->XX, your decoder uses 2*omega_L vs omega_ref.
    omega_ref = float(qd.derived.omega_ref_rad_s(TransitionPair.G_XX))  # rad/s
    omega0 = 0.5 * omega_ref  # per-photon carrier

    # --- build a Gaussian drive first (envelope + amplitude etc.) ---
    base = gaussian_field_drive(
        t0=Q(60, "ps"),
        sigma=Q(8, "ps"),
        E0=Q(4e5, "V/m"),
        energy=Q(1.3, "eV"),
        delta_omega=Q(0.0, "rad/s"),
        pol_state=None,
        preferred_kind="2ph",
        label="chirped_drive",
    )

    # --- add chirp using your existing carrier_profiles ---
    # Option A: linear chirp delta_omega(t) = rate * (t - t0)
    # rate units: rad/s^2
    delta_fn = carrier_profiles.linear_chirp(
        rate=Q(6.0e22, "rad/s^2"), t0=Q(60, "ps")
    )

    # Option B: smooth tanh chirp delta_omega(t) = delta_max * tanh((t - t0)/tau)
    # delta_fn = carrier_profiles.tanh_chirp(t0=Q(60, "ps"), delta_max=Q(5.0e11, "rad/s"), tau=Q(6, "ps"))

    # Construct Carrier so omega_L(t) = omega0 + delta_fn(t)
    # IMPORTANT: adjust this line if your Carrier signature differs.
    carrier = Carrier(omega0=Q(omega0, "rad/s"), delta_omega=delta_fn)

    drive = ClassicalFieldDriveU(
        envelope=base.envelope,
        amplitude=base.amplitude,
        carrier=carrier,
        pol_state=base.pol_state,
        pol_transform=base.pol_transform,
        preferred_kind=base.preferred_kind,
        label="chirped_linear",
    )

    # Sanity plot of omega_L(t)
    plot_drive_omega(drive, tlist=tlist, time_unit_s=time_unit_s)

    # --- run ---
    engine = SimulationEngine(audit=True)

    units = UnitSystem(time_unit_s=time_unit_s)
    specs = [DriveSpec(payload=drive, drive_id="chirped_linear")]

    bundle = qd.compile_bundle(units=units)  # placeholder, ignore
    # Use your normal pattern:
    bundle = qd.compile_bundle(
        units=__import__("smef.core.units", fromlist=["UnitSystem"]).UnitSystem(
            time_unit_s=time_unit_s
        )
    )
    dims = bundle.modes.dims()
    rho0 = rho0_qd_vacuum(dims=dims, qd_state=QDState.G)

    solve_options = {
        "method": "bdf",
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 200000,
        "max_step": 0.02,
        "progress_bar": "tqdm",
    }

    res = engine.run(
        qd,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        drives=specs,
        solve_options={"qutip_options": solve_options},
    )

    # Use your existing plot helper if you want:
    plot_qd_run_summary(res, time_unit_s=time_unit_s, drive=drive)


if __name__ == "__main__":
    main()
