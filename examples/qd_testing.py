import jax.numpy as jnp
import numpy as np
from photon_weave.operation import Operation, FockOperationType
from qutip import Options, mesolve, Qobj, mcsolve
from photon_weave.state.custom_state import CustomState
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import PolarizationLabel
from photon_weave.state.composite_envelope import CompositeEnvelope
from bec.operators.qd_operators import QDState
from bec.quantum_dot.params import CavityParams, DipoleParams, EnergyLevels
from bec.quantum_dot.qd_photon_weave import QuantumDotSystem
from photon_weave.extra import interpreter
import matplotlib.pyplot as plt

# --- Step 1: Make energy levels & QD system ---
EL = EnergyLevels(exciton=1.35, biexciton=2.7, fss=0.0)
DP = DipoleParams(dipole_moment_Cm=3.0e-29)  # ~10 Debye (1 D ≈ 3.336e-30 C·m)
CP = CavityParams(Q=1e4, Veff_um3=1.0, lambda_nm=920.0, n=3.5)
QD = QuantumDotSystem(
    EL, initial_state=QDState.G, cavity_params=CP, dipole_params=DP
)
print(f"THETA={QD.THETA}")
print(f"PHI={QD.PHI}")
print(QD.gammas)
print("-------")
print(QD.modes)
print("-------")
print(QD.context)

FLY_ENV = Envelope()
# FLY_ENV.wavelength = 442.8
FLY_ENV.wavelength = 918.4
# FLY_ENV.wavelength = 855.0

QD.register_flying_mode(FLY_ENV, label="input")

print("-----")
print(QD.modes)
print(QD._hamiltonian_light_matter_interaction())

# --- Step 2: Build optical modes / full composite space ---
ENVS = []


def op_func(dim):
    a_dag = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a_dag[n, n - 1] = 1
    return jnp.array(a_dag)


op = Operation(
    FockOperationType.Expresion,
    expr=("s_mult", 1, "a_dag"),
    context={"a_dag": lambda dims: op_func(dims[0])},
)


for mode in QD.modes:
    env_h = Envelope()
    env_h.fock.dimensions = 2
    env_v = Envelope()
    env_v.polarization.state = PolarizationLabel.V
    env_v.fock.dimensions = 2
    ce = CompositeEnvelope(env_h, env_v)
    ENVS.extend([env_h, env_v])

for i, env in enumerate(ENVS[-2:]):
    env.fock.resize(4)
    # if i == 0:
    if True:
        env.fock.apply_operation(op)


CSTATE = CompositeEnvelope(QD.dot, *ENVS)
CSTATE.combine(QD.dot, *[env.fock for env in ENVS])
CSTATE.reorder(QD.dot, *[env.fock for env in ENVS])
QD.dot.expand()

DIMENSIONS = [s.dimensions for s in [QD.dot, *[env.fock for env in ENVS]]]
DIMS = [DIMENSIONS, DIMENSIONS]
print(DIMENSIONS)

# --- Step 3: Hamiltonians ---
# FSS
H_fss_tuple = QD._hamiltonian_fss()
H_fss_qobj = Qobj(
    np.array(interpreter(H_fss_tuple, QD.context, DIMENSIONS)),
    dims=DIMS,
)

# Classical drive
H_drive_tuple = QD._hamiltonian_classical_drive()
H_drive_qobj = Qobj(
    np.array(interpreter(H_drive_tuple, QD.context, DIMENSIONS)),
    dims=DIMS,
)

H_int_qobj = Qobj(
    np.array(
        interpreter(
            QD._hamiltonian_light_matter_interaction(), QD.context, DIMENSIONS
        )
    ),
    dims=DIMS,
)

# --- Step 4: Collapse operators ---
gammas = {"L_XX_X": 1.0, "L_X_G": 1.0}
C_ops = QD.qutip_collapse_operators(DIMENSIONS)
print("Collapse Ops")
for c in C_ops:
    print(c.full().round(3))

# --- Step 5: Projectors ---
P_ops = QD.qutip_projectors(DIMENSIONS)
LM_ops = QD.qutip_light_mode_projectors(DIMENSIONS)

flying_labels = [m.label for m in QD.modes if m.source == "external"]
intrinsic_labels = [m.label for m in QD.modes if m.source == "internal"]
qd_eops = [P_ops[k] for k in ("P_G", "P_X1", "P_X2", "P_XX")]
fly_eops = [LM_ops[f"N[{lbl}]"] for lbl in flying_labels]
out_eops = [LM_ops[f"N[{lbl}]"] for lbl in intrinsic_labels]
e_ops_list = [op.to("csr") for op in (qd_eops + fly_eops + out_eops)]

idx_qd_start, idx_qd_end = 0, len(qd_eops)
idx_fly_start, idx_fly_end = idx_qd_end, idx_qd_end + len(fly_eops)
idx_out_start, idx_out_end = idx_fly_start + len(fly_eops), idx_fly_start + len(
    fly_eops
) + len(out_eops)
# --- Step 6: Initial state ---
rho0 = Qobj(np.array(CSTATE.product_states[0].state), dims=DIMS)

# ---- choose a fixed order for projectors

# (optional but recommended) keep things sparse
# e_ops_list = [op.to("csr") for op in e_ops_list]
H_fss_qobj = H_fss_qobj.to("csr")
H_drive_qobj = H_drive_qobj.to("csr")
H_int_qobj = H_int_qobj.to("csr")
C_ops = [c.to("csr") for c in C_ops]

# ---- run
t_list = np.linspace(0, 10, 500)
# Choose pulse params


def gaussian_coeff(t, args):
    # Use plain numpy for QuTiP callbacks
    dt = t - args["t0"]
    s = args["sigma"]
    return args["A"] * np.exp(-0.5 * (dt / s) ** 2)


t0 = 2.0  # center of the pulse (must lie inside your t_list)
sigma = 0.01  # pulse width
Theta = 0  # np.pi / 2  # target pulse area (pi/2 pulse)

A = Theta / (np.sqrt(2 * np.pi) * sigma)  # amplitude from desired area
args = {"t0": t0, "sigma": sigma, "A": A}
# opts = Options(nsteps=20000, rtol=1e-9, atol=1e-11, max_step=0.02 * sigma)
opts = Options(nsteps=500, rtol=1e-9, atol=1e-11, progress_bar="enhanced")
result = mesolve(
    [H_fss_qobj, H_int_qobj],  # [H_drive_qobj, gaussian_coeff]],
    rho0,
    t_list,
    c_ops=C_ops,
    # c_ops=[],
    e_ops=e_ops_list,
    args=args,
    options=opts,
)


# result.expect follows the order of e_ops_list
qd_traces = result.expect[idx_qd_start:idx_qd_end]
fly_traces = result.expect[idx_fly_start:idx_fly_end]
out_traces = result.expect[idx_out_start:idx_out_end]

t_ns = t_list  # if your time unit is already ns

fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

# Top: flying (input) modes
for lbl, y in zip(flying_labels, fly_traces):
    ax_top.plot(t_ns, y, label=f"{lbl}")
ax_top.set_ylabel("⟨N⟩ (flying)")
ax_top.legend(loc="upper right")
ax_top.grid(True, alpha=0.3)

# Middle: QD populations
qd_names = [r"$|G\rangle$", r"$|X_1\rangle$",
            r"$|X_2\rangle$", r"$|XX\rangle$"]
for name, y in zip(qd_names, qd_traces):
    ax_mid.plot(t_ns, y, label=name)
ax_mid.set_ylabel("Population")
ax_mid.legend(loc="upper right")
ax_mid.grid(True, alpha=0.3)

# Bottom: intrinsic (output) modes
for lbl, y in zip(intrinsic_labels, out_traces):
    ax_bot.plot(t_ns, y, label=f"{lbl}")
ax_bot.set_xlabel("Time (ns)")
ax_bot.set_ylabel("⟨N⟩ (output)")
ax_bot.legend(loc="upper right")
ax_bot.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
