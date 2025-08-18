from qutip import Options
import matplotlib.pyplot as plt
import numpy as np
from qutip import Options, mesolve, Qobj
from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.polarization import PolarizationLabel

from bec.light.classical import ClassicalTwoPhotonDrive
from bec.light.envelopes import GaussianEnvelope
from bec.operators.qd_operators import QDState
from bec.quantum_dot.params import CavityParams, DipoleParams, EnergyLevels
from bec.quantum_dot.qd_photon_weave import QuantumDotSystem
from bec.plots.plotter import QDPlotter

# --- Step 1: Make energy levels & QD system ---

EL = EnergyLevels(exciton=1.35, biexciton=2.7, fss=0.1)
DP = DipoleParams(dipole_moment_Cm=3.0e-29)
CP = CavityParams(Q=1e4, Veff_um3=1.0, lambda_nm=920.0, n=3.5)
QD = QuantumDotSystem(
    EL, initial_state=QDState.G, cavity_params=CP, dipole_params=DP
)


DRIVE = ClassicalTwoPhotonDrive(
    omega=GaussianEnvelope(t0=3, sigma=0.5, area=np.pi / 2), detuning=0
)

plotter = QDPlotter(QD, classical_2g=DRIVE, tlist=np.linspace(0, 10, 5000))
res = plotter.run_and_plot(filename="classical_two_photon_drive", show_top=True)
