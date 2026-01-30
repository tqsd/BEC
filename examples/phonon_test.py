from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from smef.core.drives.types import DriveSpec
from smef.engine import SimulationEngine, UnitSystem
from smef.core.units import Q

from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState, TransitionPair
from bec.quantum_dot.factories.drives import make_gaussian_field_drive_pi
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.phonon_params import (
    PhononParams,
    PhononModelType,
    PhenomenologicalPhononParams,
    PolaronPhononParams,
)


def make_qd(*, with_phonons: bool) -> QuantumDot:
    energy = EnergyStructure.from_params(
        exciton=Q(1.2, "eV"),
        binding=Q(0.2, "eV"),
        fss=Q(50e-6, "eV"),
    )
    dipoles = DipoleParams.biexciton_cascade_from_fss(
        mu_default_Cm=Q(1e-28, "C*m"),
        fss=Q(50e-6, "eV"),
    )

    if not with_phonons:
        return QuantumDot(energy=energy, dipoles=dipoles, phonons=None)

    # Reasonable starting values for a "phonons on" scenario:
    # - polaron renorm enabled (B < 1)
    # - phenomenological pure dephasing for X1/X2/XX
    # - phenomenological exciton relaxation between X1 and X2
    # - optional EID scale (set to 0.0 if you do not want drive-dependent dephasing)
    pheno = PhenomenologicalPhononParams(
        gamma_phi_Xp=Q(1.0e9, "1/s"),
        gamma_phi_Xm=Q(1.0e9, "1/s"),
        gamma_phi_XX=Q(2.0e9, "1/s"),
        gamma_relax_X1_X2=Q(1.0e8, "1/s"),
        gamma_relax_X2_X1=Q(1.0e8, "1/s"),
        gamma_phi_eid_scale=0.0,
    )
    pol = PolaronPhononParams(
        enable_polaron_renorm=True,
        alpha=Q(0.03, "s**2"),
        omega_c=Q(1.0e12, "rad/s"),
        enable_exciton_relaxation=False,
    )
    phonons = PhononParams(
        model=PhononModelType.POLARON,
        temperature=Q(4.0, "K"),
        phi_G=0.0,
        phi_X=1.0,
        phi_XX=2.0,
        phenomenological=pheno,
        polaron=pol,
    )

    return QuantumDot(energy=energy, dipoles=dipoles, phonons=phonons)


def run_case(
    *,
    qd: QuantumDot,
    engine: SimulationEngine,
    units: UnitSystem,
    tlist: np.ndarray,
    time_unit_s: float,
    label: str,
):
    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    rho0 = rho0_qd_vacuum(dims=dims, qd_state=QDState.G)

    # Two-photon resonance for G <-> XX:
    # emitter uses 2*omega_L for kind="2ph", so omega_L should be omega_ref/2
    omega_G_XX = float(qd.derived.omega_ref_rad_s(TransitionPair.G_XX))
    omega0 = 0.5 * omega_G_XX

    drive = make_gaussian_field_drive_pi(
        qd,
        pair=TransitionPair.G_XX,
        t0=Q(150.0, "ps"),
        sigma=Q(25.0, "ps"),
        preferred_kind="2ph",
        label=label,
        include_polaron=True,
        omega0_rad_s=omega0,
        chirp_rate_rad_s2=None,
    )

    specs = [DriveSpec(payload=drive, drive_id=drive.label or label)]

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

    return res, drive


def _get_expect(res, key: str) -> np.ndarray:
    if key not in res.expect:
        raise KeyError(
            "Missing expect key: %s (available: %s)"
            % (key, list(res.expect.keys()))
        )
    return np.asarray(res.expect[key], dtype=float).reshape(-1)


def main() -> None:
    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)
    units = UnitSystem(time_unit_s=time_unit_s)
    engine = SimulationEngine(audit=False)

    tlist = np.linspace(0.0, 500.0, 5001)

    qd_no = make_qd(with_phonons=False)
    qd_ph = make_qd(with_phonons=True)

    # If your drive factory already applies polaron renorm via derived, you can keep include_polaron=True.
    # If you want a strict "phonons off", set include_polaron=False in run_case for qd_no.
    res_no, drive_no = run_case(
        qd=qd_no,
        engine=engine,
        units=units,
        tlist=tlist,
        time_unit_s=time_unit_s,
        label="2ph_no_phonons",
    )

    res_ph, drive_ph = run_case(
        qd=qd_ph,
        engine=engine,
        units=units,
        tlist=tlist,
        time_unit_s=time_unit_s,
        label="2ph_with_phonons",
    )

    # Plot populations: G, X1, X2, XX for both runs
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for res, tag in [(res_no, "no phonons"), (res_ph, "with phonons")]:
        pop_G = _get_expect(res, "pop_G")
        pop_X1 = _get_expect(res, "pop_X1")
        pop_X2 = _get_expect(res, "pop_X2")
        pop_XX = _get_expect(res, "pop_XX")

        ax.plot(tlist, pop_G, label="G (%s)" % tag)
        ax.plot(tlist, pop_X1, label="X1 (%s)" % tag)
        ax.plot(tlist, pop_X2, label="X2 (%s)" % tag)
        ax.plot(tlist, pop_XX, label="XX (%s)" % tag)

    ax.set_xlabel("t (ps solver units)")
    ax.set_ylabel("population")
    ax.set_title(
        "Two-photon pi Gaussian drive: G -> XX (compare phonons on/off)"
    )
    ax.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    main()
