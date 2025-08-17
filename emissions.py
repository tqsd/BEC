import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, c, hbar, pi


# Mock data for demonstration since QuTiP is not available
# Simulate exponential decay curves for cavity and vacuum


def simulate_decay(g1, g2, g3, g4, tlist):
    pop_xx = np.exp(-(g1 + g2) * tlist)
    pop_x1 = g1 / (g1 + g2) * (np.exp(-g3 * tlist) - np.exp(-(g1 + g2) * tlist))
    pop_x2 = g2 / (g1 + g2) * (np.exp(-g4 * tlist) - np.exp(-(g1 + g2) * tlist))
    pop_g = 1 - pop_xx - pop_x1 - pop_x2
    return [pop_g, pop_x1, pop_x2, pop_xx]


# Parameters
lambda_em_nm = 930
dipole_m_C = 1e-29
lambda_m = lambda_em_nm * 1e-9
omega = 2 * pi * c / lambda_m
gamma0 = (omega**3 * dipole_m_C**2) / (3 * pi * epsilon_0 * hbar * c**3)

# Cavity parameters
Q = 10000
Veff_um3 = 0.5
Veff_m3 = Veff_um3 * 1e-18
Fp = (3 / (4 * np.pi**2)) * (lambda_m / 3.5) ** 3 * (Q / Veff_m3)
gamma_cav = gamma0 * Fp

# Time grid
tlist = np.linspace(0, 10 / gamma0, 300)

# Simulate both cases
vacuum_results = simulate_decay(gamma0, gamma0, gamma0 / 2, gamma0 / 2, tlist)
cavity_results = simulate_decay(
    gamma_cav, gamma_cav, gamma_cav / 2, gamma_cav / 2, tlist
)

# Plotting
labels = ["G", "X1", "X2", "XX"]
styles = ["-", "--", ":", "-."]

plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    plt.plot(
        tlist,
        vacuum_results[i],
        linestyle=styles[i],
        color="C0",
        label=f"{
            label} (Vacuum)",
    )
    plt.plot(
        tlist,
        cavity_results[i],
        linestyle=styles[i],
        color="C1",
        label=f"{
            label} (Cavity)",
    )

plt.xlabel("Time [s]")
plt.ylabel("Population")
plt.title("Biexciton Cascade: Vacuum vs. Cavity-Enhanced Emission")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

home_dir = os.path.expanduser("~")
output_dir = os.path.join(home_dir, ".roam", "plots")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "biexciton_cascade.png")

# Save the plot
plt.savefig(output_path)
