from matplotlib.lines import Line2D
import jax.numpy as jnp
import numpy as np
from photon_weave.operation import (
    CompositeOperationType,
    Operation,
)
from qutip import Options, mesolve, Qobj
from photon_weave.state.fock import Fock
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import PolarizationLabel
from photon_weave.state.composite_envelope import CompositeEnvelope
from bec.operators.fock_operators import (
    Ladder,
    Pol,
    rotated_ladder_operator,
)
from bec.operators.qd_operators import QDState
from bec.quantum_dot.params import CavityParams, DipoleParams, EnergyLevels
from bec.quantum_dot.qd_photon_weave import QuantumDotSystem
from photon_weave.extra import interpreter
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Step 1: Make energy levels & QD system ---
EL = EnergyLevels(exciton=1.35, biexciton=2.7, fss=0.1)
DP = DipoleParams(dipole_moment_Cm=3.0e-29)
CP = CavityParams(Q=1e4, Veff_um3=1.0, lambda_nm=920.0, n=3.5)
QD = QuantumDotSystem(
    EL, initial_state=QDState.G, cavity_params=CP, dipole_params=DP
)


# --- Step 2: Build optical modes / full composite space ---
ENVS = []


CE = CompositeEnvelope()
# ----- Constructing the space ------
for mode in QD.modes:
    print(mode)
    env_h = Envelope()
    env_h.fock.dimensions = 2
    env_v = Envelope()
    env_v.polarization.state = PolarizationLabel.V
    env_v.fock.dimensions = 2
    mode.containerHV = [env_h, env_v]
    CE = CompositeEnvelope(env_h, env_v)
    ENVS.extend([env_h, env_v])

# CE.apply_operation(op1, ENVS[-4].fock, ENVS[-3].fock)
# CE.apply_operation(op2, ENVS[-2].fock, ENVS[-1].fock)


CSTATE = CompositeEnvelope(QD.dot, *ENVS)
CSTATE.combine(QD.dot, *[env.fock for env in ENVS])
CSTATE.reorder(QD.dot, *[env.fock for env in ENVS])
QD.dot.expand()

DIMENSIONS = [s.dimensions for s in [QD.dot, *[env.fock for env in ENVS]]]
DIMS = [DIMENSIONS, DIMENSIONS]

t_list = np.linspace(0, 10, 500)

# --- Step 3: Hamiltonians ---
# FSS
H_fss = QD.qutip_hamiltonian_fss(dims=DIMENSIONS).to(
    "csr"
)  # or Qobj(interpreter(...))
H = [H_fss]


def gaussian_area_to_amp(theta, g, sigma):
    # area theta = ∫ g*A*exp(-(t-t0)^2/(2σ^2)) dt = g*A*σ*sqrt(2π)
    return theta / (g * sigma * np.sqrt(2 * np.pi))


def gaussian_coeff_factory(A, t0, sigma):
    A = float(A)
    t0 = float(t0)
    sigma = float(sigma)
    inv2s2 = 0.5 / (sigma * sigma)

    def coeff(t, args=None):  # <- args is optional
        dt = t - t0
        return A * np.exp(-inv2s2 * dt * dt)  # returns scalar

    return coeff


H = [QD.qutip_hamiltonian_fss(dims=DIMENSIONS).to("csr")]

H_drive = QD.qutip_hamiltonian_classical_drive(DIMENSIONS).to("csr")
Theta = np.pi / 2  # π-pulse area (for H = Ω(t) * (|G><XX| + h.c.))
t0 = 2.0
sigma = 0.5
g = 1.0

A = gaussian_area_to_amp(Theta, g, sigma)
H.append([g * H_drive, gaussian_coeff_factory(A, t0, sigma)])

gammas = {"L_XX_X": 1.0, "L_X_G": 1.0}
C_ops = QD.qutip_collapse_operators(DIMENSIONS)


# --- Step 5: Projectors ---
P_ops = QD.qutip_projectors(DIMENSIONS)
LM_ops = QD.qutip_light_mode_projectors(DIMENSIONS)

# Which basis index does each projector pick out?
for name in ("P_G", "P_X1", "P_X2", "P_XX"):
    block = P_ops[name].ptrace(0).full().diagonal().real
    print(name, "→ basis index", int(np.argmax(block)))
# Also print the enum ints used to set your initial state:
print(
    "QDState enum:",
    int(QDState.G),
    int(QDState.X1),
    int(QDState.X2),
    int(QDState.XX),
)
flying_labels = [m.label for m in QD.modes if m.source == "external"]
intrinsic_labels = [m.label for m in QD.modes if m.source == "internal"]
intrinsic_plot_labels = [
    f"{m.wavelength_nm:.1f}" for m in QD.modes if m.source == "internal"
]
qd_eops = [P_ops[k] for k in ("P_G", "P_X1", "P_X2", "P_XX")]
# polarization-resolved (minus=H, plus=V)
fly_eops_total = [LM_ops[f"N[{lbl}]"] for lbl in flying_labels]
out_eops_total = [LM_ops[f"N[{lbl}]"] for lbl in intrinsic_labels]
fly_eops_H = [LM_ops[f"N-[{lbl}]"] for lbl in flying_labels]  # H
fly_eops_V = [LM_ops[f"N+[{lbl}]"] for lbl in flying_labels]  # V
out_eops_H = [LM_ops[f"N-[{lbl}]"] for lbl in intrinsic_labels]
out_eops_V = [LM_ops[f"N+[{lbl}]"] for lbl in intrinsic_labels]

# (optional) single-photon projectors in each pol
fly_P10 = [
    LM_ops.get(f"P10[{lbl}]")
    for lbl in flying_labels
    if f"P10[{lbl}]" in LM_ops
]
fly_P01 = [
    LM_ops.get(f"P01[{lbl}]")
    for lbl in flying_labels
    if f"P01[{lbl}]" in LM_ops
]

# build the evaluation list (order matters; remember indices)
e_ops_list = [
    op.to("csr")
    for op in (
        qd_eops
        + fly_eops_total
        + fly_eops_H
        + fly_eops_V
        + out_eops_total
        + out_eops_H
        + out_eops_V
        # + (fly_P10 or []) + (fly_P01 or [])  # if you added them
    )
]

i0 = 0
idx_qd = slice(i0, i0 + len(qd_eops))
i0 += len(qd_eops)
idx_fly_T = slice(i0, i0 + len(fly_eops_total))
i0 += len(fly_eops_total)
idx_fly_H = slice(i0, i0 + len(fly_eops_H))
i0 += len(fly_eops_H)
idx_fly_V = slice(i0, i0 + len(fly_eops_V))
i0 += len(fly_eops_V)
idx_out_T = slice(i0, i0 + len(out_eops_total))
i0 += len(out_eops_total)
idx_out_H = slice(i0, i0 + len(out_eops_H))
i0 += len(out_eops_H)
idx_out_V = slice(i0, i0 + len(out_eops_V))
i0 += len(out_eops_V)

# --- Step 6: Initial state  for QuTiP ---
rho0 = Qobj(np.array(CSTATE.product_states[0].state), dims=DIMS).to("csr")
rho_qd0 = rho0.ptrace(0)  # subsystem 0 = QD
print("QD populations at t=0:", np.real_if_close(rho_qd0.diag()))

opts = Options(nsteps=500, rtol=1e-9, atol=1e-11, progress_bar="enhanced")
result = mesolve(
    H,
    rho0,
    t_list,
    c_ops=C_ops,
    # c_ops=[],
    e_ops=e_ops_list,
    options=opts,
)


# result.expect follows the order of e_ops_list
qd_traces = result.expect[idx_qd]
fly_T_traces = result.expect[idx_fly_T]
fly_H_traces = result.expect[idx_fly_H]
fly_V_traces = result.expect[idx_fly_V]
out_T_traces = result.expect[idx_out_T]
out_H_traces = result.expect[idx_out_H]
out_V_traces = result.expect[idx_out_V]
t_ns = t_list  # if your time unit is already ns


# --- styling choices ---
# QD state colors (distinct!)
QD_COLORS = {
    r"$|G\rangle$": "#1f77b4",  # blue
    r"$|X_1\rangle$": "#ff7f0e",  # orange
    r"$|X_2\rangle$": "#2ca02c",  # green
    r"$|XX\rangle$": "#d62728",  # red
}

QD_SHAPES = {
    r"$|G\rangle$": "-",  # blue
    r"$|X_1\rangle$": "--",  # orange
    r"$|X_2\rangle$": "-.",  # green
    r"$|XX\rangle$": "-",  # red
}
# --- figure size for a single column ---
COL_W_CM, H_CM = 8.6, 5.0
cm = 1 / 2.54
fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
    3,
    1,
    figsize=(COL_W_CM * cm, H_CM * cm),
    sharex=True,
    constrained_layout=True,
)

# --- compact styles for small figures ---
mpl.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
LW, ALPHA = 1.3, 0.95
LS_H, LS_V = "-", "--"  # H solid, V dashed

# ---------- TOP: classical drive (since no flying inputs) ----------
Omega = g * A * np.exp(-0.5 * ((t_ns - t0) / sigma) ** 2)

ax_top.plot(t_ns, Omega, lw=LW, alpha=ALPHA, label=r"$\Omega(t)$")
ax_top.set_ylabel(r"$\Omega(t)$")

# (optional) show the accumulated area on a right y-axis and mark π (or Θ)
area = np.trapz(Omega, t_ns)
ax_top2 = ax_top.twinx()
ax_top2.plot(
    t_ns,
    np.cumsum(Omega) * (t_ns[1] - t_ns[0]),
    ls="--",
    lw=LW,
    alpha=0.7,
    label=r"$\int^t \Omega dt$",
)
ax_top2.set_ylabel(r"area (rad)")


# Add a tiny style key instead of a full legend
# ---------- MID: QD populations ----------

qd_names = [r"$|G\rangle$", r"$|X_1\rangle$",
            r"$|X_2\rangle$", r"$|XX\rangle$"]
for name, y in zip(qd_names, qd_traces):
    ax_mid.plot(
        t_ns,
        y,
        color=QD_COLORS[name],
        ls=QD_SHAPES[name],
        lw=LW,
        alpha=ALPHA,
        label=name,
    )
ax_mid.set_ylabel("QD")
ax_mid.legend(
    loc="upper right", ncol=2, framealpha=0.8, borderpad=0.2, handlelength=1.2
)

# ---------- BOT: outputs ----------
for m_idx, (lbl, Ht, Vt) in enumerate(
    zip(intrinsic_labels, out_H_traces, out_V_traces)
):
    color = f"C{m_idx}"
    ax_bot.plot(t_ns, Ht, color=color, ls=LS_H, lw=LW, alpha=ALPHA)
    ax_bot.plot(t_ns, Vt, color=color, ls=LS_V, lw=LW, alpha=ALPHA)
    ax_bot.text(
        t_ns[-1],
        Ht[-1],
        f"",
        va="center",
        ha="left",
        fontsize=6,
        color=color,
    )
ax_bot.set_xlabel("Time (ns)")
ax_bot.set_ylabel(r"$\langle N\rangle$ (out)")

# keep ticks sparse for height 5 cm
for ax in (ax_top, ax_mid, ax_bot):
    ax.locator_params(axis="y", nbins=3)
    ax.locator_params(axis="x", nbins=5)
    ax.grid(True, alpha=0.25)

fig.savefig("classical_drive.pdf", bbox_inches="tight")
fig.savefig("classical_drive.png", dpi=600, bbox_inches="tight")
