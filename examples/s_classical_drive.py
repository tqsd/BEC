import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from photon_weave.extra import interpreter
from photon_weave.state.custom_state import CustomState
from bec.operators.qd_operators import QDState, transition_operator

from qutip import basis, Qobj, mesolve


QD = CustomState(4)
QD.expand()
QD.expand()
print(QD.state)
h_bar = 1.054571817e-34
h_bar = 1

context = {
    "sigma_G_XX": lambda dims: jnp.array(
        transition_operator(QDState.G, QDState.XX)
    ),
    "sigma_XX_G": lambda dims: jnp.array(
        transition_operator(QDState.XX, QDState.G)
    ),
}

H_drive = ("s_mult", h_bar, ("add", "sigma_XX_G", "sigma_G_XX"))

H_drive = Qobj(jnp.array(interpreter(H_drive, context, dimensions=[4])))


def Omega(t, args):
    return float(
        args["amp"]
        * jnp.exp(-((t - args["t0"]) ** 2) / (2 * args["sigma"] ** 2))
    )


print(H_drive)

rho_qutip = Qobj(np.array(QD.state))
psi0 = basis(4, 0)

t_list = jnp.linspace(5, 15, 2000)
sigma = 1
amp = np.sqrt(np.pi / 2) / sigma * 0.5

args = {
    "amp": amp,
    "t0": 10.0,
    "sigma": sigma,
}
Omega_vals = np.array([Omega(t, args) for t in t_list])
pulse_area = np.trapz(Omega_vals, t_list)
print("Pulse area:", pulse_area)

observables = [
    basis(4, 0) * basis(4, 0).dag(),  # |G⟩⟨G|
    basis(4, 3) * basis(4, 3).dag(),
]  # |XX⟩⟨XX|

H_qutip = [[H_drive, Omega]]
result = mesolve(H_qutip, psi0, t_list, [], observables, args=args)

plt.plot(t_list, result.expect[0], label=r"$|G\rangle$ population")
plt.plot(t_list, result.expect[1], label=r"$|XX\rangle$ population")

plt.xlabel("Time (arb. units)")
plt.ylabel("Population")
plt.title(
    r"Rabi Oscillations: Classical Drive $|G\rangle \leftrightarrow |XX\rangle$"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
