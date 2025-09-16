from scipy.constants import e as _e, hbar as _hbar, pi, c as _c
import numpy as np
import matplotlib.pyplot as plt

from bec.channel.factory import ChannelFactory
from bec.channel.io import ChannelIO
from bec.helpers.pprint import pretty_density
from bec.light.envelopes import GaussianEnvelope, TabulatedEnvelope
from bec.operators.qd_operators import QDState
from bec.plots.quick import plot_traces
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.helpers import infer_index_sets_from_registry
from bec.simulation.engine import SimulationEngine, SimulationConfig
from bec.simulation.scenarios import ClassicalDriveScenario

# device/physics params
from bec.params.energy_levels import EnergyLevels
from bec.params.cavity_params import CavityParams
from bec.params.dipole_params import DipoleParams

# classical 2γ drive
from bec.light.classical import ClassicalTwoPhotonDrive
from bec.simulation.solvers import QutipMesolveBackend, MesolveOptions


def main():
    # -----------------------------
    # 1) Energy levels (in eV)
    # -----------------------------
    # Example numbers typical for InGaAs QDs; tweak for your sample:
    EXCITON_E = 1.300  # exciton center (eV)
    FSS = 0  # 1e-6  # fine-structure splitting 10 μeV = 1e-5 eV
    DELTA_PRIME = 0.0  # anisotropic mixing term (eV), set if needed
    # Set biexciton such that binding energy is ~3 meV: E_bind ≈ (X1 + X2) - E_XX
    X1 = EXCITON_E + 0.5 * FSS
    X2 = EXCITON_E - 0.5 * FSS
    BINDING = 3e-3  # 3 meV
    E_XX = (X1 + X2) - BINDING

    EL = EnergyLevels(
        biexciton=E_XX,
        exciton=EXCITON_E,
        fss=FSS,
        delta_prime=DELTA_PRIME,
    )

    # -----------------------------
    # 2) Photonic environment (Purcell) + dipole
    # -----------------------------
    CP = CavityParams(
        Q=5.0e4,  # moderate Q
        Veff_um3=0.5,  # effective mode volume (μm^3)
        lambda_nm=930.0,  # around 930 nm
        n=3.4,  # refractive index
    )

    # Dipole moment ~ 10 Debye (e D ≈ 3.33564e-30 C·m)
    DP = DipoleParams(dipole_moment_Cm=10.0 * 3.33564e-30)

    # -----------------------------
    # 3) Build the QD façade
    # -----------------------------
    qd = QuantumDot(
        EL,
        cavity_params=CP,
        dipole_params=DP,
        time_unit_s=1e-9,
        initial_state=QDState.G,
    )
    # scalar overlap(s)
    print("late |Λ| =", qd.diagnostics.effective_overlap("late"))
    print("early |Λ| =", qd.diagnostics.effective_overlap("early"))

    # full summary
    # -----------------------------
    # 4) Classical two-photon drive
    # -----------------------------
    # physical pulse params
    sigma = 1e-12  # s
    t0 = 1e-9  # s
    omega0 = 2e9  # rad/s

    w_XXG = float(EL.XX) * _e / _hbar
    detuning = 10e10
    wl = 0.5 * w_XXG + detuning

    pulse_area = np.pi / omega0 * 0.80

    # tabulate a normalized Gaussian (window ±nsigma*sigma)
    nsigma = 6.0
    num = 1001
    t = np.linspace(t0 - nsigma * sigma, t0 + nsigma * sigma, num)
    norm = pulse_area / (sigma * np.sqrt(2 * np.pi))
    y = norm * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    # (optional) renormalize area because of truncation + discretization
    trap = np.trapezoid(y, t)
    if trap > 0:
        y *= pulse_area / trap

    # build the envelope and drive
    env_tab = TabulatedEnvelope(t=tuple(map(float, t)), y=tuple(map(float, y)))
    drive = ClassicalTwoPhotonDrive(
        envelope=env_tab,
        omega0=omega0,
        detuning=0.0,
        label="2g",
        laser_omega=wl,
    )

    # (optional) JSON round-trip
    # {'type': 'classical_2photon', 'envelope': {'type': 'tabulated', ...}, ...}
    drive_json = drive.to_dict()
    drive = ClassicalTwoPhotonDrive.from_dict(drive_json)

    scenario = ClassicalDriveScenario(drive=drive)
    # -----------------------------
    # 5) Simulation config & engine
    # -----------------------------
    tlist = np.linspace(0.0, 2.0, 1001)  # 0 → 10 ns, 501 points
    # {0,1} photons per pol
    cfg = SimulationConfig(tlist=tlist, trunc_per_pol=2, time_unit_s=1e-9)

    engine = SimulationEngine()

    backend = QutipMesolveBackend(
        MesolveOptions(
            nsteps=10000,
            rtol=1e-9,
            atol=1e-11,
            progress_bar="tqdm",
            store_final_state=True,
        )
    )
    engine = SimulationEngine(solver=backend)

    # -----------------------------
    # 6) Run + collect traces
    # -----------------------------
    traces, rho_final, rho_phot_final = engine.run_with_state(qd, scenario, cfg)

    print(qd.diagnostics.mode_layout_summary(rho_phot=rho_phot_final))
    factory = ChannelFactory()
    src = factory.from_photonic_state_prepare_from_scalar(rho_phot_final)

    ChannelIO.save_npz("biexciton_source.npz", src)
    infer_index_sets_from_registry(qd)

    early, late, plus_set, minus_set, dims, offset = (
        infer_index_sets_from_registry(qd, rho_has_qd=False)
    )
    print(pretty_density(rho_phot_final, dims))

    fig = plot_traces(
        traces,
        title=r"Biexciton classical 2$\gamma$ drive (Gaussian)",
        save=None,  # e.g., "biexciton_2g.png"
    )
    plt.show()


if __name__ == "__main__":
    main()
