import numpy as np
import jax.numpy as jnp
from qutip import Qobj, mesolve, basis, expect, tensor, concurrence
import matplotlib.pyplot as plt

from photon_weave.extra import interpreter
from photon_weave.state.custom_state import CustomState
from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope

from bec.operators.fock_operators import rotated_ladder_operator
from bec.operators.qd_operators import QDState, transition_operator

# Create Quantum Dot
QD = CustomState(4)
QD.state = 1
QD.expand()
QD.expand()
print(QD)


# Create the optical states
env0_h = Envelope()
env0_v = Envelope()

env1_h = Envelope()
env1_v = Envelope()

for env in [env0_v, env0_h, env1_h, env1_v]:
    env.fock.dimensions = 2
    env.expand()
    env.expand()

CENV0 = CompositeEnvelope(env0_h, env0_v)
CENV0.combine(env0_v.fock, env0_h.fock)
CENV1 = CompositeEnvelope(env1_h, env1_v)
CENV1.combine(env1_v.fock, env1_h.fock)


STATE = CompositeEnvelope(QD, CENV0, CENV1)
STATE.combine(QD, env0_h.fock, env1_h.fock)
STATE.reorder(QD, env0_h.fock, env0_v.fock, env1_h.fock, env1_v.fock)
print(STATE.product_states[0].state.shape)


Delta = 0.1  # diagonal split
Delta_p = 0.00  # set >0 to turn on mixing


def exciton_rotation_params(delta, delta_p):
    # Hamiltonian in |X1>, |X2> basis
    H = 0.5 * np.array([[delta, delta_p], [delta_p, -delta]], dtype=complex)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(H)

    # First eigenvector defines rotation (up to global phase)
    v = eigvecs[:, 0]
    theta = np.arctan2(abs(v[1]), abs(v[0]))
    phi = np.angle(v[1]) - np.angle(v[0])

    return theta, phi, eigvals


# Example: delta=0.01, delta_p=0.00 (pure diagonal split)
THETA, PHI, E = exciton_rotation_params(Delta, Delta_p)
context = {
    # Decay operators
    "s_XX_X1": lambda dims: jnp.array(
        transition_operator(QDState.XX, QDState.X1)
    ),
    "s_XX_X2": lambda dims: jnp.array(
        transition_operator(QDState.XX, QDState.X2)
    ),
    "s_X1_G": lambda dims: jnp.array(
        transition_operator(QDState.X1, QDState.G)
    ),
    "s_X2_G": lambda dims: jnp.array(
        transition_operator(QDState.X2, QDState.G)
    ),
    # Excitation operators
    "s_G_X1": lambda dims: jnp.array(
        transition_operator(QDState.G, QDState.X1)
    ),
    "s_G_X2": lambda dims: jnp.array(
        transition_operator(QDState.G, QDState.X2)
    ),
    "s_X1_XX": lambda dims: jnp.array(
        transition_operator(QDState.X1, QDState.XX)
    ),
    "s_X2_XX": lambda dims: jnp.array(
        transition_operator(QDState.X2, QDState.XX)
    ),
    # Quantum Dot identity
    "iqd": lambda dims: jnp.eye(4),
    # Light mode operators
    "a0+": lambda dims: rotated_ladder_operator(
        dims[1], THETA, PHI, operator="annihilation"
    ),
    "a0+_dag": lambda dims: rotated_ladder_operator(
        dims[1], THETA, PHI, operator="creation"
    ),
    "n0+": lambda dims: rotated_ladder_operator(dims[1], THETA)
    @ rotated_ladder_operator(dims[1], THETA, operator="annihilation"),
    "a0-": lambda dims: rotated_ladder_operator(
        dims[1], THETA, PHI, mode="minus", operator="annihilation"
    ),
    "a0-_dag": lambda dims: rotated_ladder_operator(
        dims[1], THETA, PHI, mode="minus", operator="creation"
    ),
    "n0-": lambda dims: rotated_ladder_operator(dims[1], THETA, mode="minus")
    @ rotated_ladder_operator(
        dims[1], THETA, PHI, mode="minus", operator="annihilation"
    ),
    "a1+": lambda dims: rotated_ladder_operator(
        dims[3], THETA, PHI, operator="annihilation"
    ),
    "a1+_dag": lambda dims: rotated_ladder_operator(
        dims[3], THETA, PHI, operator="creation"
    ),
    "n1+": lambda dims: rotated_ladder_operator(dims[3], THETA)
    @ rotated_ladder_operator(dims[3], THETA, operator="annihilation"),
    "a1-": lambda dims: rotated_ladder_operator(
        dims[3], THETA, PHI, mode="minus", operator="annihilation"
    ),
    "a1-_dag": lambda dims: rotated_ladder_operator(
        dims[3], THETA, PHI, mode="minus", operator="creation"
    ),
    "n1-": lambda dims: rotated_ladder_operator(dims[3], THETA, mode="minus")
    @ rotated_ladder_operator(
        dims[3], THETA, PHI, mode="minus", operator="annihilation"
    ),
    "i0": lambda dims: jnp.eye(dims[1] * dims[2]),
    "i1": lambda dims: jnp.eye(dims[3] * dims[4]),
}
# Hamiltonian
dimensions = [4] + [
    env.fock.dimensions for env in [env0_h, env0_v, env1_h, env1_v]
]


def get_number_operator(op1: str, op2: str):
    return Qobj(
        np.array(interpreter(("kron", "iqd", op1, op2), context, dimensions)),
        dims=[dimensions, dimensions],
    )


N0_plus = get_number_operator("n0+", "i1")
N0_minus = get_number_operator("n0-", "i1")
N1_plus = get_number_operator("i0", "n1+")
N1_minus = get_number_operator("i0", "n1-")

proj_X1 = transition_operator(QDState.X1, QDState.X1)
proj_X2 = transition_operator(QDState.X2, QDState.X2)
X1X2 = transition_operator(QDState.X1, QDState.X2)
X2X1 = transition_operator(QDState.X2, QDState.X1)

H_fss_local = (Delta / 2) * (proj_X1 - proj_X2) + (Delta_p / 2) * (X1X2 + X2X1)
H_fss = interpreter(
    ("kron", jnp.array(H_fss_local), "i0", "i1"), context, dimensions
)
H_fss = Qobj(np.array(H_fss), dims=[dimensions, dimensions])


A = interpreter(("kron", "iqd", "a0+", "i1"), context, dimensions)
Ad = interpreter(("kron", "iqd", "a0+_dag", "i1"), context, dimensions)
print(
    "Adjoint ok? ", np.allclose(np.array(Ad), np.array(A).conj().T, atol=1e-12)
)

# Collapse operators
collapse_ops = {
    "L_XX_X": (
        "add",
        ("kron", "s_XX_X1", "a0+_dag", "i1"),
        ("kron", "s_XX_X2", "a0-_dag", "i1"),
    ),
    "L_X_G": (
        "add",
        ("kron", "s_X1_G", "i0", "a1-_dag"),
        ("kron", "s_X2_G", "i0", "a1+_dag"),
    ),
}
print("a0+_dag:\n", rotated_ladder_operator(2, jnp.pi / 2))
gamma = 1.0
C_ops = []
for key in collapse_ops:
    op = interpreter(collapse_ops[key], context, dimensions)
    C_ops.append(
        Qobj(np.array(jnp.sqrt(gamma) * op), dims=[dimensions, dimensions])
    )


rho = Qobj(
    np.array(STATE.product_states[0].state), dims=[dimensions, dimensions]
)
print("Initial population in |XX⟩: ", rho[3, 3])

# Check collapse operators one by one
for i, L in enumerate(C_ops):
    print(f"Collapse operator {i}: norm = {L.norm()}")
    print(f"⟨ψ|L†L|ψ⟩ = ", (L.dag() * L * rho).tr())

t_list = np.linspace(0, 10, 200)
result = mesolve(H_fss, rho, t_list, C_ops, [])

qd_rhos = [state.ptrace(0) for state in result.states]

pop_G = [rho[0, 0].real for rho in qd_rhos]
pop_X1 = [rho[1, 1].real for rho in qd_rhos]
pop_X2 = [rho[2, 2].real for rho in qd_rhos]
pop_XX = [rho[3, 3].real for rho in qd_rhos]


plt.plot(t_list, pop_G, label="|G⟩")
plt.plot(t_list, pop_X1, label="|X1⟩")
plt.plot(t_list, pop_X2, label="|X2⟩")
plt.plot(t_list, pop_XX, label="|XX⟩")

plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Quantum Dot State Populations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


n0_plus_vals = expect(N0_plus, result.states)
n0_minus_vals = expect(N0_minus, result.states)
n1_plus_vals = expect(N1_plus, result.states)
n1_minus_vals = expect(N1_minus, result.states)
plt.plot(t_list, n0_plus_vals, label=r"$n_{0,+}$")
plt.plot(t_list, n0_minus_vals, label=r"$n_{0,-}$")
plt.plot(t_list, n1_plus_vals, label=r"$n_{1,+}$")
plt.plot(t_list, n1_minus_vals, label=r"$n_{1,-}$")
plt.xlabel("Time")
plt.ylabel("Photon Number")
plt.title("Photon Numbers in Polarization Modes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Take final state and trace out QD ---
rho_final = result.states[-1]
rho_light = rho_final.ptrace([1, 2, 3, 4])  # keep (H0, V0, H1, V1)

# --- Build single-photon basis states for each bin ---
ket1, ket0 = basis(2, 1), basis(2, 0)

# Bin 0 (H0, V0)
H0 = tensor(ket1, ket0)
V0 = tensor(ket0, ket1)

# Bin 1 (H1, V1)
H1 = tensor(ket1, ket0)
V1 = tensor(ket0, ket1)

# Full 4-mode kets
HH = tensor(H0, H1)
HV = tensor(H0, V1)
VH = tensor(V0, H1)
VV = tensor(V0, V1)

# Project to exactly 1 photon in each bin
P_11 = HH * HH.dag() + HV * HV.dag() + VH * VH.dag() + VV * VV.dag()
rho_11 = P_11 * rho_light * P_11
p_11 = float(rho_11.tr())
rho_11 = rho_11 / p_11

# --- Bell fidelities in HV basis ---
phi_plus_HV = (HH + VV).unit()
phi_minus_HV = (HH - VV).unit()
psi_plus_HV = (HV + VH).unit()
psi_minus_HV = (HV - VH).unit()

F_phi_plus_HV = float(expect(phi_plus_HV.proj(), rho_11))
F_phi_minus_HV = float(expect(phi_minus_HV.proj(), rho_11))
F_psi_plus_HV = float(expect(psi_plus_HV.proj(), rho_11))
F_psi_minus_HV = float(expect(psi_minus_HV.proj(), rho_11))

print("Prob(1 photon/bin) =", p_11)
print("F(Φ+_HV) =", F_phi_plus_HV)
print("F(Φ-_HV) =", F_phi_minus_HV)
print("F(Ψ+_HV) =", F_psi_plus_HV)
print("F(Ψ-_HV) =", F_psi_minus_HV)

# --- RL basis version ---
R0 = (H0 + 1j * V0).unit()
L0 = (H0 - 1j * V0).unit()
R1 = (H1 + 1j * V1).unit()
L1 = (H1 - 1j * V1).unit()

RR = tensor(R0, R1)
RL = tensor(R0, L1)
LR = tensor(L0, R1)
LL = tensor(L0, L1)

phi_plus_RL = (RR + LL).unit()  # Φ+ in RL
psi_plus_RL = (RL + LR).unit()  # Ψ+ in RL

F_phi_plus_RL = float(expect(phi_plus_RL.proj(), rho_11))
F_psi_plus_RL = float(expect(psi_plus_RL.proj(), rho_11))

print("F(Φ+_RL) =", F_phi_plus_RL)
print("F(Ψ+_RL) =", F_psi_plus_RL)

# --- Sanity check: populations and HH–VV coherence ---
pops_HV = [float((k.dag() * rho_11 * k)) for k in [HH, HV, VH, VV]]
coh_HH_VV = (HH.dag() * rho_11 * VV)[0, 0]
print("HV pops =", pops_HV)
print("|<HH|rho|VV>| =", abs(coh_HH_VV))
