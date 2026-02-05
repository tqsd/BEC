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
from bec.light.core.polarization import JonesState
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


def make_qd() -> QuantumDot:
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

    energy = EnergyStructure(
        X1=Q(1.150, "eV"),
        X2=Q(1.250, "eV"),
        XX=Q(2.600, "eV"),
    )
    energy = EnergyStructure.from_params(
        exciton=Q(1.2, "eV"), binding=Q(0.2, "eV"), fss=Q(0e-6, "eV")
    )
    dipoles = DipoleParams(mu_default=Q(1e-28, "C*m"))
    dipoles = DipoleParams.biexciton_cascade_from_fss(
        mu_default_Cm=Q(1e-29, "C*m"), fss=Q(0e-6, "eV")
    )
    return QuantumDot(energy=energy, dipoles=dipoles)  # , phonons=phonons)


def make_resonant_carrier(*, omega_target_rad_s: float) -> Carrier:
    return Carrier(
        omega0=Q(float(omega_target_rad_s), "rad/s"),
        delta_omega=carrier_profiles.constant(Q(0.0, "rad/s")),
    )


def make_gaussian_drive(
    *,
    label: str,
    omega_target_rad_s: float,
    t0_ps: float,
    sigma_ps: float,
    E0_v_m: float,
    preferred_kind: str,
    pol_state=None,
) -> ClassicalFieldDriveU:
    base = gaussian_field_drive(
        t0=Q(float(t0_ps), "ps"),
        sigma=Q(float(sigma_ps), "ps"),
        E0=Q(float(E0_v_m), "V/m"),
        energy=Q(1.3, "eV"),
        delta_omega=Q(0.0, "rad/s"),
        pol_state=pol_state,
        preferred_kind=preferred_kind,
        label=label,
    )

    carrier = make_resonant_carrier(omega_target_rad_s=omega_target_rad_s)

    return ClassicalFieldDriveU(
        envelope=base.envelope,
        amplitude=base.amplitude,
        carrier=carrier,
        pol_state=base.pol_state,
        pol_transform=base.pol_transform,
        preferred_kind=base.preferred_kind,
        label=label,
    )


def _find_h_term_index(problem, label_substr: str) -> int:
    for i, t in enumerate(problem.h_terms):
        if getattr(t, "label", "") and label_substr in t.label:
            return i
    return -1


def _debug_compiled_problem(problem, *, drive_label_substr: str) -> None:
    if problem.rho0 is None:
        raise ValueError("problem.rho0 is None")

    rho0 = np.asarray(problem.rho0, dtype=complex)
    diag = np.real(np.diag(rho0))
    idx0 = int(np.argmax(diag))
    p0 = float(diag[idx0])

    print("D:", int(problem.D), "dims:", tuple(problem.dims))
    print("rho0 dominant basis index:", idx0, "prob:", p0)

    i_drive = _find_h_term_index(problem, drive_label_substr)
    if i_drive < 0:
        print("Could not find H term containing:", drive_label_substr)
        print("H labels:", [t.label for t in problem.h_terms])
        return

    term = problem.h_terms[i_drive]
    op = np.asarray(term.op, dtype=complex)

    row = np.abs(op[idx0, :])
    col = np.abs(op[:, idx0])

    max_row = float(row.max())
    max_col = float(col.max())
    j_row = int(row.argmax())
    j_col = int(col.argmax())

    print("drive term index:", i_drive, "label:", term.label)
    print("max |op[idx0, :]|:", max_row, "at j:", j_row)
    print("max |op[:, idx0]|:", max_col, "at i:", j_col)

    if max_row == 0.0 and max_col == 0.0:
        print(
            "WARNING: drive operator does not connect the initial basis state."
        )
        print("This will produce no dynamics regardless of the coeff(t).")


def _debug_expect(res, keys=("pop_G", "pop_X1", "pop_X2", "pop_XX")) -> None:
    for k in keys:
        if k not in res.expect:
            continue
        arr = np.asarray(res.expect[k], dtype=float)
        print(k, "min:", float(arr.min()), "max:", float(arr.max()))


def main() -> None:
    qd = make_qd()
    print(qd.derived.report())

    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)
    units = UnitSystem(time_unit_s=time_unit_s)

    # Use audit=False here because we want to do our own checks between compile and run.
    engine = SimulationEngine(audit=True)

    tlist = np.linspace(0.0, 500.0, 5001)

    omega_G_X1 = float(qd.derived.omega_ref_rad_s(TransitionPair.G_X1))

    drive_H = make_gaussian_drive(
        label="H1",
        omega_target_rad_s=omega_G_X1,
        t0_ps=100.0,
        sigma_ps=50.0,
        E0_v_m=1e7,
        preferred_kind="1ph",
        pol_state=JonesState.H(),
    )

    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    rho0 = rho0_qd_vacuum(dims=dims, qd_state=QDState.G)

    specs = [DriveSpec(payload=drive_H, drive_id=drive_H.label)]

    # Step 1: compile only
    problem = engine.compile(
        qd,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        drives=specs,
    )

    # Step 2: check compiled problem before solving
    _debug_compiled_problem(problem, drive_label_substr="H_drive_H1")
    # If you want, also print full audit text here:
    # from smef.core.sim.audit import audit_problem_dense
    # print(audit_problem_dense(problem))

    # Step 3: solve
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

    res = (
        engine.adapter.solve(problem, options=solve_options)
        if engine.adapter
        else SimulationEngine(audit=True).run(
            qd,
            tlist=tlist,
            time_unit_s=time_unit_s,
            rho0=rho0,
            drives=specs,
            solve_options=solve_options,
        )
    )

    _debug_expect(res)

    figs = plot_runs(
        [res],
        units=units,
        drives=[drive_H],
        qds=[qd],
        cfg=PlotConfig(
            title="Single scenario: H pulse (G->X1)",
            ncols=1,
            column_titles=["H pulse (G->X1)"],
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
