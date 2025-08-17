from photon_weave.state.envelope import Envelope
from bec.operators.qd_operators import QDState
from bec.quantum_dot.params import CavityParams, DipoleParams, EnergyLevels
from bec.quantum_dot.qd_photon_weave import QuantumDotSystem

# --- Step 1: Make energy levels & QD system ---

EL = EnergyLevels(exciton=1.35, biexciton=2.7, fss=0.1)
DP = DipoleParams(dipole_moment_Cm=3.0e-29)
CP = CavityParams(Q=1e4, Veff_um3=1.0, lambda_nm=920.0, n=3.5)
QD = QuantumDotSystem(
    EL, initial_state=QDState.G, cavity_params=CP, dipole_params=DP
)

FLY_ENV_1 = Envelope()
FLY_ENV_1.wavelength = 918.4

QD.register_flying_mode(FLY_ENV_1, label="first")
