from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from smef.core.drives.types import DriveSpec
from smef.core.units import Q
from smef.engine import SimulationEngine, UnitSystem

from bec.light.classical import carrier_profiles
from bec.light.classical.carrier import Carrier
from bec.light.classical.factories import gaussian_field_drive
from bec.light.classical.field_drive import ClassicalFieldDriveU
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState, TransitionPair
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.reporting.plotting.api import plot_run
from bec.reporting.plotting.grid import PlotConfig


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
        E0=Q(2e4, "V/m"),
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
    # plot_qd_run_summary(res, time_unit_s=time_unit_s, drive=drive)
    fig = plot_run(
        res,
        units=units,
        drive=drive,
        qd=qd,
        cfg=PlotConfig(
            title="Chirped drive",
            show_omega_L=True,
            show_coupling_panel=True,
            coupling_mode="abs",
        ),
    )
    plt.show()


if __name__ == "__main__":
    main()
