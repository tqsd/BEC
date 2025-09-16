r"""
Example: Biexciton excitation of a quantum dot with a Gaussian two-photon
drive.

We simulate a four-level quantum dot (G, X1, X2, XX) driven by a
classical two-photon pulse. The Hamiltonian includes fine-structure splitting,
cavity Purcell enhancement, and radiative decays.

Mathematics
-----------
Laser drive:
    \Omega(t) = \Omega_0 f(t)
where f(t) is a Gaussian envelope.

Laser frequency:
    \omega_L = \tfrac{1}{2} \omega_{XX \to G} + \Delta
with detuning \Delta.

Simulation units:
    all rates \gamma [1/s] are converted by multiplying with `time_unit_s`.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e as _e, hbar as _hbar, pi, c as _c

# BEC components
from bec.channel.factory import ChannelFactory
from bec.channel.io import ChannelIO
from bec.helpers.pprint import pretty_density
from bec.light.envelopes import GaussianEnvelope
from bec.light.classical import ClassicalTwoPhotonDrive
from bec.operators.qd_operators import QDState
from bec.plots.quick import plot_traces
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.helpers import infer_index_sets_from_registry
from bec.simulation.engine import SimulationEngine, SimulationConfig
from bec.simulation.scenarios import ClassicalDriveScenario
from bec.simulation.solvers import QutipMesolveBackend, MesolveOptions

# Device parameters
from bec.params.energy_levels import EnergyLevels
from bec.params.cavity_params import CavityParams
from bec.params.dipole_params import DipoleParams


def detuning_to_wavelength_nm(EL, detuning=0.0):
    """Return laser wavelength (nm) for given biexciton–ground energy and detuning."""
    w_xxg = float(EL.XX) * _e / _hbar
    wl = 0.5 * w_xxg + detuning
    return (2 * pi * _c / wl) * 1e9


def main():
    # -----------------------------
    # 1) Energy levels (in eV)
    # -----------------------------
    exciton_e = 1.300  # exciton center
    fss = 0.0  # fine-structure splitting \Delta
    delta_p = 0.0  # anisotropic mixing \Delta'
    binding = 3e-3  # biexciton binding energy (~3 meV)

    x1 = exciton_e + 0.5 * fss
    x2 = exciton_e - 0.5 * fss
    e_xx = (x1 + x2) - binding

    EL = EnergyLevels(
        biexciton=e_xx,
        exciton=exciton_e,
        fss=fss,
        delta_prime=delta_p,
    )

    # -----------------------------
    # 2) Photonic environment + dipole
    # -----------------------------
    CP = CavityParams(Q=5e4, Veff_um3=0.5, lambda_nm=930.0, n=3.4)
    DP = DipoleParams(dipole_moment_Cm=10.0 * 3.33564e-30)  # ~10 Debye

    # -----------------------------
    # 3) Quantum dot system
    # -----------------------------
    qd = QuantumDot(
        EL,
        cavity_params=CP,
        dipole_params=DP,
        time_unit_s=1e-9,
        initial_state=QDState.G,
    )

    print("late |Lambda| =", qd.diagnostics.effective_overlap("late"))
    print("early |Lambda| =", qd.diagnostics.effective_overlap("early"))

    # -----------------------------
    # 4) Classical two-photon drive
    # -----------------------------
    sigma = 1e-11  # pulse width (s)
    t0 = 1e-9  # pulse center (s)
    omega0 = 1e10  # Rabi amplitude (rad/s)

    w_xxg = float(EL.XX) * _e / _hbar
    detuning = 5e10
    wL = 0.5 * w_xxg + detuning

    pulse_area = np.pi / omega0
    lam_nm = detuning_to_wavelength_nm(EL, detuning=detuning)
    print(f"Laser wavelength = {lam_nm:.2f} nm")

    env = GaussianEnvelope(t0=t0, sigma=sigma, area=pulse_area)
    drive = ClassicalTwoPhotonDrive(
        envelope=env,
        omega0=omega0,
        detuning=0.0,
        label="2g",
        laser_omega=wL,
    )
    scenario = ClassicalDriveScenario(drive=drive)

    # -----------------------------
    # 5) Simulation config & engine
    # -----------------------------
    tlist = np.linspace(0.0, 20.0, 20001)  # 0 → 20 ns
    cfg = SimulationConfig(tlist=tlist, trunc_per_pol=2, time_unit_s=1e-10)

    backend = QutipMesolveBackend(
        MesolveOptions(
            nsteps=20000,
            rtol=1e-9,
            atol=1e-11,
            progress_bar="tqdm",
            store_final_state=True,
            max_step=1e-2,
        )
    )
    engine = SimulationEngine(solver=backend)

    # -----------------------------
    # 6) Run simulation
    # -----------------------------
    traces, rho_final, rho_phot_final = engine.run_with_state(qd, scenario, cfg)

    print(qd.diagnostics.mode_layout_summary(rho_phot=rho_phot_final))

    factory = ChannelFactory()
    src = factory.from_photonic_state_prepare_from_scalar(rho_phot_final)
    ChannelIO.save_npz("biexciton_source.npz", src)

    early, late, plus_set, minus_set, dims, offset = (
        infer_index_sets_from_registry(qd, rho_has_qd=False)
    )
    print("Resulting state:")
    print(pretty_density(rho_phot_final, dims))

    fig = plot_traces(
        traces,
        title="Biexciton classical two-photon drive (Gaussian)",
    )
    plt.show()


if __name__ == "__main__":
    main()
