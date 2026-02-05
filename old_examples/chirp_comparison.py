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
from bec.quantum_dot.spec.phonon_params import (
    PhenomenologicalPhononParams,
    PhononModelType,
    PhononParams,
    PolaronPhononParams,
)
from bec.reporting.plotting.api import plot_runs
from bec.reporting.plotting.grid import PlotConfig

phonons = PhononParams(
    model=PhononModelType.POLARON,
    temperature=Q(4.0, "K"),
    phenomenological=PhenomenologicalPhononParams(
        gamma_phi_Xp=Q(1.0e9, "1/s"),
        gamma_phi_Xm=Q(1.0e9, "1/s"),
        gamma_phi_XX=Q(0.0, "1/s"),
    ),
    polaron=PolaronPhononParams(
        enable_polaron_renorm=True,
        alpha=Q(0.03, "ps**2"),
        omega_c=Q(1.0e12, "rad/s"),
        enable_exciton_relaxation=False,
    ),
)


def make_drive_pair_tanh(
    *, qd: QuantumDot
) -> tuple[ClassicalFieldDriveU, ClassicalFieldDriveU]:
    omega_ref = float(qd.derived.omega_ref_rad_s(TransitionPair.G_XX))
    omega0 = 0.5 * omega_ref

    base = gaussian_field_drive(
        t0=Q(60, "ps"),
        sigma=Q(8, "ps"),
        E0=Q(2e4, "V/m"),
        energy=Q(1.3, "eV"),
        delta_omega=Q(0.0, "rad/s"),
        pol_state=None,
        preferred_kind="2ph",
        label="base",
    )

    carrier_plain = Carrier(
        omega0=Q(omega0, "rad/s"),
        delta_omega=carrier_profiles.constant(Q(0.0, "rad/s")),
    )
    drive_plain = ClassicalFieldDriveU(
        envelope=base.envelope,
        amplitude=base.amplitude,
        carrier=carrier_plain,
        pol_state=base.pol_state,
        pol_transform=base.pol_transform,
        preferred_kind=base.preferred_kind,
        label="plain",
    )

    delta_fn = carrier_profiles.tanh_chirp(
        t0=Q(60, "ps"),
        delta_max=Q(5.0e10, "rad/s"),
        tau=Q(6.0, "ps"),
    )
    carrier_chirp = Carrier(
        omega0=Q(omega0, "rad/s"),
        delta_omega=delta_fn,
    )
    drive_chirp = ClassicalFieldDriveU(
        envelope=base.envelope,
        amplitude=base.amplitude,
        carrier=carrier_chirp,
        pol_state=base.pol_state,
        pol_transform=base.pol_transform,
        preferred_kind=base.preferred_kind,
        label="tanh_chirp",
    )

    return drive_plain, drive_chirp


def run_once(
    *,
    engine: SimulationEngine,
    qd: QuantumDot,
    drive: ClassicalFieldDriveU,
    tlist: np.ndarray,
    time_unit_s: float,
    units: UnitSystem,
):
    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    rho0 = rho0_qd_vacuum(dims=dims, qd_state=QDState.G)

    specs = [DriveSpec(payload=drive, drive_id=drive.label)]

    solve_options = {
        "qutip_options": {
            "method": "bdf",
            "atol": 1e-10,
            "rtol": 1e-8,
            "nsteps": 200000,
            "max_step": 0.02,
            "progress_bar": "tqdm",
        }
    }

    return engine.run(
        qd,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        drives=specs,
        solve_options=solve_options,
    )


def main() -> None:
    energy = EnergyStructure(
        X1=Q(1.201, "eV"),
        X2=Q(1.201, "eV"),
        XX=Q(2.600, "eV"),
    )
    dipoles = DipoleParams(mu_default=Q(1e-27, "C*m"))

    qd = QuantumDot(energy=energy, dipoles=dipoles, phonons=phonons)

    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)
    tlist = np.linspace(0.0, 200.0, 2001)

    units = UnitSystem(time_unit_s=time_unit_s)
    engine = SimulationEngine(audit=True)

    drive_plain, drive_chirp = make_drive_pair_tanh(qd=qd)

    res_plain = run_once(
        engine=engine,
        qd=qd,
        drive=drive_plain,
        tlist=tlist,
        time_unit_s=time_unit_s,
        units=units,
    )
    res_chirp = run_once(
        engine=engine,
        qd=qd,
        drive=drive_chirp,
        tlist=tlist,
        time_unit_s=time_unit_s,
        units=units,
    )

    figs = plot_runs(
        [res_plain, res_chirp],
        units=units,
        drives=[drive_plain, drive_chirp],
        qds=[qd, qd],
        cfg=PlotConfig(
            title="Plain vs tanh chirp (phonons enabled)",
            ncols=2,
            column_titles=["Plain", "Tanh chirp"],
            show_omega_L=True,
            show_coupling_panel=True,
            coupling_mode="abs",
            sharex=True,
            sharey_by_row=True,
        ),
    )

    for fig in figs:
        fig.show()

    plt.show()


if __name__ == "__main__":
    main()
