from __future__ import annotations

# from bec.quantum_dot.smef.drives.pipeline import QDDriveDecodeContext
import matplotlib.pyplot as plt
import numpy as np
from smef.core.drives.types import DriveSpec
from smef.core.units import Q, UnitSystem
from smef.engine import SimulationEngine

from bec.light.classical.factories import gaussian_field_drive
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure


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


def plot_expectations(res, *, time_unit_s: float) -> None:
    # time axis in ps for readability
    t_solver = np.asarray(getattr(res, "tlist", None))
    if t_solver.size == 0:
        # fallback: some engines store it on the problem; adjust if needed
        raise AttributeError(
            "Result has no tlist; pass it explicitly if needed."
        )
    t_ps = (t_solver * float(time_unit_s)) * 1e12

    expect = getattr(res, "expect", None)
    if expect is None or len(expect) == 0:
        print("No expectations recorded (res.expect is empty).")
        return

    def _get(name: str) -> np.ndarray:
        y = np.asarray(expect[name], dtype=float)
        if y.shape[0] != t_ps.shape[0]:
            raise ValueError(
                f"Trace {name} has length {
                    y.shape[0]} but t has {t_ps.shape[0]}"
            )
        return y

    # --- QD populations ---
    pop_keys = ["pop_G", "pop_X1", "pop_X2", "pop_XX"]
    pop_keys = [k for k in pop_keys if k in expect]

    if pop_keys:
        plt.figure()
        for k in pop_keys:
            plt.plot(t_ps, _get(k), label=k)
        plt.xlabel("t (ps)")
        plt.ylabel("population")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.title("QD state populations")
        plt.tight_layout()
        plt.show()

    # --- Photon numbers ---
    n_keys = ["n_GX_H", "n_GX_V", "n_XX_H", "n_XX_V"]
    n_keys = [k for k in n_keys if k in expect]

    if n_keys:
        plt.figure()
        for k in n_keys:
            plt.plot(t_ps, _get(k), label=k)
        plt.xlabel("t (ps)")
        plt.ylabel("mean photon number")
        plt.legend()
        plt.title("Photon numbers in output modes")
        plt.tight_layout()
        plt.show()

    # --- Sanity check: total population (should be ~1 if no leakage) ---
    if all(k in expect for k in ["pop_G", "pop_X1", "pop_X2", "pop_XX"]):
        total = _get("pop_G") + _get("pop_X1") + \
            _get("pop_X2") + _get("pop_XX")
        plt.figure()
        plt.plot(t_ps, total, label="pop_total")
        plt.xlabel("t (ps)")
        plt.ylabel("sum populations")
        plt.title("Population conservation check")
        plt.tight_layout()
        plt.show()


def main() -> None:
    # Build a minimal spec (adapt constructors to your real ones)
    energy = EnergyStructure(
        X1=Q(1.201, "eV"),
        X2=Q(1.201, "eV"),
        XX=Q(2.600, "eV"),
    )
    dipoles = DipoleParams(
        mu_default=Q(1e-27, "C*m")
    )  # your real dipole params

    qd = QuantumDot(energy=energy, dipoles=dipoles)

    # Build one drive (your real drive type)

    drive = gaussian_field_drive(
        t0=Q(30, "ps"),
        sigma=Q(10, "ps"),
        E0=Q(4e5, "V/m"),
        energy=Q(1.3, "eV"),  # or omega0=..., or wavelength=...
        delta_omega=Q(0.0, "rad/s"),
        pol_state=None,
        preferred_kind="2ph",
        label="test_drive",
    )

    # Simulation settings
    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)
    tlist = np.linspace(0.0, 200.0, 401)  # solver units

    engine = SimulationEngine(audit=True)

    # IMPORTANT: supply drive_id explicitly so we can map id -> payload
    specs = [DriveSpec(payload=drive, drive_id="test_drive")]

    # Inject drive_id -> payload mapping into decode context
    # engine will call this normally; here for access

    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)
    units = UnitSystem(time_unit_s=time_unit_s)
    bundle = qd.compile_bundle(units=units)

    problem = engine.compile(
        qd,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=None,
        drives=specs,
    )

    print("num c_terms:", len(problem.c_terms))
    for i, t in enumerate(problem.c_terms):
        print(i, t.label, t.meta)

    # If you want to actually solve, add rho0 and run:

    dims = bundle.modes.dims()
    rho0 = rho0_qd_vacuum(dims=dims, qd_state=QDState.G)

    solve_options = {
        "method": "bdf",  # stiff-safe; "adams" can be fine but bdf is more robust here
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 200000,
        "max_step": 0.02,  # solver units; for sigma=3ps and time_unit=1ps this is conservative
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
    plot_qd_run_summary(res, time_unit_s=time_unit_s, drive=drive)


if __name__ == "__main__":
    main()
