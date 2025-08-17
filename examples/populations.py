from bec.quantum_dot.qd_qutip import QuantumDotSystemQuTiP
from bec.quantum_dot.params import EnergyLevels
from bec.plots.qd_ensemble import QuantumDotEnsembleConfig, QuantumDotEnsemble

cfg = QuantumDotEnsembleConfig(
    label="Vacuum decay",
    energy_levels=EnergyLevels(base=1e14, fss=1e9, delta_prime=0.0),
    initial_state=None,  # Defaults to |G⟩⟨G|
    input_modes=None,  # No photons entering
    sim_params=dict(Omega0=1.0, t0=5.0, sigma=1.5),
)

# 👇 Pass the class, not an instance
plotter = QuantumDotEnsemble(system=QuantumDotSystemQuTiP, configs=[cfg])
path = plotter.plot("qdot_ensemble_plot.png")

print("Plot saved at:", path)
