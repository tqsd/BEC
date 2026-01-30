from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from smef.core.drives.types import DriveSpec
from smef.engine import SimulationEngine, UnitSystem
from smef.core.units import Q

from bec.metrics.metrics import QDDiagnostics
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState, TransitionPair
from bec.quantum_dot.factories.drives import make_gaussian_field_drive_pi
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.dipole_params import DipoleParams

from bec.reporting.plotting.api import plot_runs
from bec.reporting.plotting.grid import PlotConfig


def make_qd() -> QuantumDot:
    fss = Q(50e-6, "eV")
    energy = EnergyStructure.from_params(
        exciton=Q(1.2, "eV"),
        binding=Q(0.2, "eV"),
        fss=fss,
    )
    dipoles = DipoleParams.biexciton_cascade_from_fss(
        mu_default_Cm=Q(1e-28, "C*m"),
        fss=fss,
    )
    return QuantumDot(energy=energy, dipoles=dipoles)


def main() -> None:
    qd = make_qd()
    print(qd.derived.report())

    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)
    units = UnitSystem(time_unit_s=time_unit_s)

    engine = SimulationEngine(audit=True)

    tlist = np.linspace(0.0, 600.0, 6001)

    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    rho0 = rho0_qd_vacuum(dims=dims, qd_state=QDState.G)

    # ------------------------------------------------------------------
    # Bichromatic ladder scheme: G -> X1 (pump color) and X1 -> XX (stokes color)
    # Two *different* carrier frequencies are inferred automatically from the pair.
    #
    # Timing: sequential pi pulses (pump first, then stokes).
    # ------------------------------------------------------------------

    sigma = Q(12.0, "ps")  # narrower than 40 ps (more bandwidth)
    t_pump = Q(150.0, "ps")
    t_stokes = Q(210.0, "ps")  # delayed so population can move stepwise

    drive_pump = make_gaussian_field_drive_pi(
        qd,
        pair=TransitionPair.G_X1,
        t0=t_pump,
        sigma=sigma,
        preferred_kind="1ph",
        label="pump_GX1_pi",
        include_polaron=True,
        # omega0_rad_s omitted -> inferred as omega_ref(G<->X1)
        # pol_state omitted -> auto from dipole
    )

    drive_stokes = make_gaussian_field_drive_pi(
        qd,
        pair=TransitionPair.X1_XX,
        t0=t_stokes,
        sigma=sigma,
        preferred_kind="1ph",
        label="stokes_X1XX_pi",
        include_polaron=True,
        # omega0 inferred as omega_ref(X1<->XX)
    )

    # If you find systematic overshoot, scale both together a bit:
    # drive_pump = drive_pump.scaled(0.95)
    # drive_stokes = drive_stokes.scaled(0.95)

    specs = [
        DriveSpec(payload=drive_pump, drive_id=drive_pump.label or "pump"),
        DriveSpec(
            payload=drive_stokes, drive_id=drive_stokes.label or "stokes"
        ),
    ]

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

    res = engine.run(
        qd,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        drives=specs,
        solve_options=solve_options,
    )

    metrics = QDDiagnostics().compute(qd, res, units=units)
    print(metrics.to_text())

    figs = plot_runs(
        [res],
        units=units,
        drives=[None],  # or [drive_pump] if you only want to visualize one
        qds=[qd],
        cfg=PlotConfig(
            title="Bichromatic ladder",
            ncols=1,
            column_titles=["bichromatic ladder"],
            show_omega_L=False,  # since we didn't provide a single drive
            show_coupling_panel=False,  # same reason
            sharex=True,
            sharey_by_row=True,
        ),
    )
    for fig in figs:
        fig.show()
    plt.show()


if __name__ == "__main__":
    main()
