r"""
Biexciton excitation with a symbolic (JSON) two-photon drive.

Math (LaTeX, for docs only, not used in code):
  Drive envelope:      \Omega(t) = \Omega_0 f(t)
  Two-photon laser:    \omega_L = \tfrac{1}{2}\,\omega_{XX\to G} + \Delta
  Envelope (Gaussian): f(t) = A \exp\big(-(t-t_0)^2/(2\sigma^2)\big)

Notes
-----
- All solver times are dimensionless; `time_unit_s` sets seconds per solver unit.
- Spontaneous rates are scaled by `time_unit_s` inside the decay model.
- Using a symbolic/tabulated envelope introduces Python callback overhead.
  For speed, prefer an analytic envelope class if available.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e as _e, hbar as _hbar, c as _c, pi

# project imports
from bec.channel.factory import ChannelFactory
from bec.channel.io import ChannelIO
from bec.helpers.pprint import pretty_density
from bec.operators.qd_operators import QDState
from bec.plots.quick import plot_traces
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.helpers import infer_index_sets_from_registry
from bec.simulation.engine import SimulationEngine, SimulationConfig
from bec.simulation.scenarios import ClassicalDriveScenario
from bec.simulation.solvers import QutipMesolveBackend, MesolveOptions

# device/physics params
from bec.params.energy_levels import EnergyLevels
from bec.params.cavity_params import CavityParams
from bec.params.dipole_params import DipoleParams

# classical two-photon drive
from bec.light.classical import ClassicalTwoPhotonDrive


def detuning_to_wavelength_nm(EL, detuning=0.0):
    """Return laser wavelength (nm) for given biexciton–ground energy and detuning."""
    w_xxg = float(EL.XX) * _e / _hbar
    wl = 0.5 * w_xxg + detuning
    return (2 * pi * _c / wl) * 1e9


def main():
    # -----------------------------
    # 1) Energy levels (in eV)
    # -----------------------------
    exciton_e = 1.300  # exciton center (eV)
    fss = 1e-6  # fine-structure splitting (~1 micro-eV)
    delta_prime = 0.0  # anisotropic mixing term (eV)
    binding = 3e-3  # biexciton binding energy (~3 meV)

    x1 = exciton_e + 0.5 * fss
    x2 = exciton_e - 0.5 * fss
    e_xx = (x1 + x2) - binding

    EL = EnergyLevels(
        biexciton=e_xx,
        exciton=exciton_e,
        fss=fss,
        delta_prime=delta_prime,
    )

    # -----------------------------
    # 2) Photonic environment (Purcell) + dipole
    # -----------------------------
    CP = CavityParams(
        Q=5.0e4,  # moderate Q
        Veff_um3=0.5,  # effective mode volume (um^3)
        lambda_nm=930.0,
        n=3.4,
    )
    DP = DipoleParams(dipole_moment_Cm=10.0 * 3.33564e-30)  # about 10 Debye

    # -----------------------------
    # 3) Quantum dot facade
    # -----------------------------
    # If not provided, QuantumDot defaults to time_unit_s=1e-9
    qd = QuantumDot(
        EL,
        cavity_params=CP,
        dipole_params=DP,
        initial_state=QDState.G,
    )

    print("late |Lambda| =", qd.diagnostics.effective_overlap("late"))
    print("early |Lambda| =", qd.diagnostics.effective_overlap("early"))

    # -----------------------------
    # 4) Classical two-photon drive
    # -----------------------------
    # QuTiP will pass dimensionless time t'; we interpret physical time as:
    #   t_phys = time_unit_s * t'.
    # Here time_unit_s = 1e-9 s (set in SimulationConfig below).

    sigma = 1e-10  # seconds
    t0 = 1e-9  # seconds
    omega0 = 2e9  # rad/s (Rabi amplitude)

    # Choose envelope amplitude so that integral of f(t) matches desired area.
    # For a normalized Gaussian with width sigma, int f(t) dt = A * sigma * sqrt(2*pi).
    # If you want area = pi/omega0 * 0.80, then A = (pi / omega0) * 0.80 / (sigma * sqrt(2*pi)).
    time_unit_s = 1e-9  # must match SimulationConfig

    # target solver-area: 0.8*pi
    # need physical area = (0.8*pi) / (time_unit_s * omega0)
    area = (np.pi * 0.8) / (time_unit_s * omega0)

    # for f(t) = A * exp(-(t-t0)^2/(2*sigma^2)),  ∫ f dt = A * sigma * sqrt(2*pi)
    A = float(area / (sigma * np.sqrt(2.0 * np.pi)))

    # Two-photon laser frequency around half the XX->G frequency, plus detuning.
    w_xxg = float(EL.XX) * _e / _hbar  # rad/s
    detuning = 1.0e11  # rad/s
    wL = 0.5 * w_xxg + detuning

    # Symbolic envelope specified as JSON (your driver supports this)
    env_json = {
        "type": "symbolic",
        "expr": "A*np.exp(-(t-t0)**2/(2*sigma**2))",
        "params": {"A": A, "t0": t0, "sigma": sigma},
    }

    drive = ClassicalTwoPhotonDrive.from_envelope_json(
        env_json,
        omega0=omega0,
        detuning=0.0,  # set to detuning if you want an explicit Hamiltonian detuning term
        label="2gamma",
        laser_omega=wL,
    )
    # round-trip (optional)
    drive = ClassicalTwoPhotonDrive.from_dict(drive.to_dict())

    scenario = ClassicalDriveScenario(drive=drive)

    # -----------------------------
    # 5) Simulation config and engine
    # -----------------------------
    # Use the same time_unit_s here and in the QuantumDot (its default is 1e-9).
    time_unit_s = 1e-9
    # Simulate from 0 to 2 ns with 1001 points:
    # solver units, since time_unit_s=1e-9, this is 0..2 ns
    tlist = np.linspace(0.0, 2.0, 1001)

    cfg = SimulationConfig(
        tlist=tlist,
        trunc_per_pol=2,  # {0,1} photons per pol
        time_unit_s=time_unit_s,
    )

    backend = QutipMesolveBackend(
        MesolveOptions(
            nsteps=10000,
            rtol=1e-9,
            atol=1e-11,
            progress_bar="tqdm",
            store_final_state=True,
            max_step=1e-2,  # cap internal step; big speedup with time-dependent coeffs
        )
    )
    engine = SimulationEngine(solver=backend)

    # -----------------------------
    # 6) Run and collect
    # -----------------------------
    traces, rho_final, rho_phot_final = engine.run_with_state(qd, scenario, cfg)

    print(qd.diagnostics.mode_layout_summary(rho_phot=rho_phot_final))

    factory = ChannelFactory()
    src = factory.from_photonic_state_prepare_from_scalar(rho_phot_final)
    ChannelIO.save_npz("biexciton_source.npz", src)

    early, late, plus_set, minus_set, dims, offset = (
        infer_index_sets_from_registry(qd, rho_has_qd=False)
    )

    print("Resulting state")
    print(pretty_density(rho_phot_final, dims))

    fig = plot_traces(
        traces,
        title=r"Biexciton classical two-photon drive (Gaussian)",
        save=None,
    )
    plt.show()


if __name__ == "__main__":
    main()
