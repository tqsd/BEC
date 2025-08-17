import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, mesolve, Options, tensor

from bec.quantum_dot.qd_qutip import QuantumDotSystem
from bec.light.lightmode_qutip import LightModeQuTiP
from bec.quantum_dot.params import EnergyLevels
from bec.operators.qd_operators import QDState

# --- Energy levels (ideal values, no FSS) ---
energy_levels = EnergyLevels(
    base=2e9, fss=0.0, delta_prime=0  # Hz  # ideal case
)
qd = QuantumDotSystem(energy_levels=energy_levels)

# --- Light modes for two decay stages ---
light1 = LightModeQuTiP(
    wavelength_nm=920.0,
    fock_dim=2,
    rotation_angle_rad=0.0,
    polarization_mode="plus",
    coupling_strength_Hz=1e6,
)
light2 = LightModeQuTiP(
    wavelength_nm=930.0,
    fock_dim=2,
    rotation_angle_rad=0.0,
    polarization_mode="plus",
    coupling_strength_Hz=1e6,
)

# --- Full Hilbert space ---
space = [qd, light1, light2]

# --- Time evolution range ---
tlist = np.linspace(0, 10e-9, 300)

# --- Build Hamiltonian ---
H = qd.build_hamiltonian(tlist=tlist, space=space)

# --- Initial state: |XX, 0, 0⟩
fock_dim = light1.fock_dim

vac1 = tensor(basis(fock_dim, 0), basis(fock_dim, 0))  # Fock1_H ⊗ Fock1_V
vac2 = tensor(basis(fock_dim, 0), basis(fock_dim, 0))  # Fock2_H ⊗ Fock2_V

qd_XX = basis(qd.dim, QDState.XX.value)

psi0 = tensor(qd_XX, vac1, vac2)
# --- Collapse operators (not yet implemented in this snippet, but assume available) ---
c_ops = []  # or: qd.build_collapse_operators(space)

# --- Observables (just QD populations for now) ---
observables = qd.get_observables()

# --- Solve time evolution ---
result = mesolve(
    H, psi0, tlist, c_ops, observables, options=Options(nsteps=5000)
)

# --- Plot ---
plt.figure(figsize=(8, 5))
for i, label in enumerate(["G", "X1", "X2", "XX"]):
    plt.plot(tlist * 1e9, result.expect[i], label=label)

plt.title("Ideal QD Biexciton Cascade")
plt.xlabel("Time [ns]")
plt.ylabel("Population")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
