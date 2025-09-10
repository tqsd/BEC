from math import sqrt, pi
import numpy as np
import matplotlib.pyplot as plt

# --- your project imports (paths match what you showed) ---
from bec.operators.qd_operators import QDState
from bec.plots.quick import plot_traces
from bec.quantum_dot.dot import QuantumDot
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
    FSS = 10e-6  # fine-structure splitting 10 μeV = 1e-5 eV
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

    # Dipole moment ~ 10 Debye (1 D ≈ 3.33564e-30 C·m)
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
    # -----------------------------
    # 4) Classical two-photon drive
    # -----------------------------
    # QuTiP will pass *dimensionless* time t that we interpret in ns here
    t0 = 0.5  # pulse center (ns)
    sigma = 0.04  # pulse width (ns)

    def omega0_for_area(sigma, area=np.pi / 2) -> float:
        return float(area / (sigma * sqrt(2 * pi)))

    Omega0 = omega0_for_area(sigma, area=np.pi / 2)  # ≈ 1.566

    def gaussian_rabi(t, _args=None):
        return Omega0 * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    drive = ClassicalTwoPhotonDrive(
        omega=gaussian_rabi,
        detuning=0.0,
        label="2g",
    )
    scenario = ClassicalDriveScenario(drive=drive)

    # -----------------------------
    # 5) Simulation config & engine
    # -----------------------------
    tlist = np.linspace(0.0, 2.0, 1001)  # 0 → 10 ns, 501 points
    # {0,1} photons per pol
    cfg = SimulationConfig(tlist=tlist, trunc_per_pol=2)

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
    print(rho_phot_final)

    # -----------------------------
    # 7) Inspect results
    # -----------------------------
    print("=== Simulation summary ===")
    print(f"classical drive?     : {traces.classical}")
    print(f"flying input labels  : {traces.flying_labels}")
    print(f"intrinsic out labels : {traces.intrinsic_labels}")
    print(f"t axis (ns)          : {traces.t.shape} points")

    # Final QD populations (P_G, P_X1, P_X2, P_XX at final time)
    P_G, P_X1, P_X2, P_XX = [arr[-1] for arr in traces.qd]
    print("\nFinal QD populations:")
    print(f"  P_G  = {P_G:.4f}")
    print(f"  P_X1 = {P_X1:.4f}")
    print(f"  P_X2 = {P_X2:.4f}")
    print(f"  P_XX = {P_XX:.4f}")
    print("max P_X2:", float(np.max(P_X2)))

    # If classical, show peak Ω(t) and pulse area
    if traces.classical and traces.omega is not None:
        print("\nClassical panel:")
        print(f"  max Ω(t) = {np.max(traces.omega):.4f}")
        if traces.area is not None:
            print(f"  final pulse area ≈ {traces.area[-1]:.4f}")

    # Example: total photon numbers on the first intrinsic output mode (if present)
    if traces.intrinsic_labels:
        print("\nExample output mode:", traces.intrinsic_labels[0])
        # The corresponding traces are in out_H/out_V with matching order of intrinsic_labels
        # Here we just show maxima across all intrinsic modes as a quick glance:
        print(
            f"  max N_H (any out) = {
                max([np.max(x) for x in traces.out_H]) if traces.out_H else 0.0:.4e}"
        )
        print(
            f"  max N_V (any out) = {
                max([np.max(x) for x in traces.out_V]) if traces.out_V else 0.0:.4e}"
        )

    fig = plot_traces(
        traces,
        title=r"Biexciton classical 2$\gamma$ drive (Gaussian)",
        save=None,  # e.g., "biexciton_2g.png"
    )
    plt.show()


if __name__ == "__main__":
    main()
