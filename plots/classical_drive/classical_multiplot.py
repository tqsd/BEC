import numpy as np
from bec.light.classical import ClassicalTwoPhotonDrive
from bec.light.envelopes import GaussianEnvelope
from bec.operators.qd_operators import QDState
from bec.plots.multiplot import plot_qd_comparison
from bec.quantum_dot.params import CavityParams, DipoleParams, EnergyLevels
from bec.quantum_dot.qd_photon_weave import QuantumDotSystem
from bec.plots.plotter import QDPlotter

# --- Step 1: Make energy levels & QD system ---
T = 15.0
EL = EnergyLevels(exciton=1.35, biexciton=2.7, fss=0.1)
DP = DipoleParams(dipole_moment_Cm=3.0e-29)
CP = CavityParams(Q=1e4, Veff_um3=1.0, lambda_nm=920.0, n=3.5)
QD1 = QuantumDotSystem(
    EL, initial_state=QDState.G, cavity_params=CP, dipole_params=DP
)

QD2 = QuantumDotSystem(
    EL, initial_state=QDState.G, cavity_params=CP, dipole_params=DP
)


DRIVE1 = ClassicalTwoPhotonDrive(
    omega=GaussianEnvelope(t0=3, sigma=1, area=np.pi / 2), detuning=0
)

sigma2 = 100 * T  # very wide → looks flat over [0,T]
Omega0 = 0.6  # target amplitude
area2 = Omega0 * np.sqrt(2 * np.pi) * sigma2
DRIVE2 = ClassicalTwoPhotonDrive(
    omega=GaussianEnvelope(t0=T / 2, sigma=sigma2, area=area2), detuning=0
)

plotter1 = QDPlotter(QD1, classical_2g=DRIVE1, tlist=np.linspace(0, T, 10000))
plotter2 = QDPlotter(QD2, classical_2g=DRIVE2, tlist=np.linspace(0, T, 5000))


def debug_drive(plotter):
    cf = plotter.classical_2g.qutip_coeff()  # callable f(t, args)
    t = plotter.tlist
    Om = np.array([cf(tt, {}) for tt in t])
    print("Ω(t): max =", Om.max(), " area ≈", np.trapezoid(Om, t))
    return Om


_ = debug_drive(plotter1)
_ = debug_drive(plotter2)

fig = plot_qd_comparison(
    [plotter1, plotter2],
    titles=["Short", "Long"],
    show_top=True,
    filename="multiplot_classical",
    figsize=(9.2, 2.5),
)
