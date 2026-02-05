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
from bec.metrics.metrics import QDDiagnostics
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
    units = UnitSystem(time_unit_s=time_unit_s)

    # --- set per-photon omega0 for 2ph G<->XX ---
    omega_ref = float(qd.derived_view.omega_ref_rad_s(TransitionPair.G_XX))
    omega0 = 0.5 * omega_ref

    # --- build drive ---
    base = gaussian_field_drive(
        t0=Q(60, "ps"),
        sigma=Q(3, "ps"),
        E0=Q(8e5, "V/m"),
        energy=Q(1.3, "eV"),
        delta_omega=Q(0.0, "rad/s"),
        pol_state=None,
        preferred_kind="2ph",
        label="drive_base",
    )

    carrier = Carrier(
        omega0=Q(omega0, "rad/s"),
    )

    drive = ClassicalFieldDriveU(
        envelope=base.envelope,
        amplitude=base.amplitude,
        carrier=carrier,
        pol_state=base.pol_state,
        pol_transform=base.pol_transform,
        preferred_kind=base.preferred_kind,
        label="chirped_linear",
    )

    # --- quick sanity plot of omega_L(t) on the same grid ---
    t_phys = (time_unit_s * tlist).astype(float)
    omega_L = np.array(
        [float(drive.omega_L_rad_s(float(t))) for t in t_phys], dtype=float
    )

    # --- initial state ---
    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    rho0 = rho0_qd_vacuum(dims=dims, qd_state=QDState.G)

    # --- run ---
    engine = SimulationEngine(audit=True)
    specs = [DriveSpec(payload=drive, drive_id="chirped_linear")]

    qutip_options = {
        "method": "bdf",
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 20000,
        "max_step": 0.02,
        "progress_bar": "tqdm",
    }

    res = engine.run(
        qd,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        drives=specs,
        solve_options={"qutip_options": qutip_options},
    )
    diag = QDDiagnostics()
    m = diag.compute(qd, res, units=units)
    print(m.to_text(precision=6))

    fig = plot_run(
        res,
        units=units,
        drives=[drive],
        qd=qd,
    )
    plt.show()


if __name__ == "__main__":
    main()
