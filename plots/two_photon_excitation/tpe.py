import jax.numpy as jnp
import numpy as np
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.operation import (
    CompositeOperationType,
    Operation,
)

from bec.operators.fock_operators import (
    Ladder,
    Pol,
    rotated_ladder_operator,
)
from bec.light.envelopes import GaussianEnvelope
from bec.operators.qd_operators import QDState
from bec.plots.multiplot import plot_qd_comparison
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
FLY_ENV_1 = Envelope()
FLY_ENV_1.wavelength = 916.17

gaussian = GaussianEnvelope(t0=2, sigma=0.5, area=np.pi / 2)

tlist = np.linspace(0, 10, 500)
ft = np.array([gaussian(tt) for tt in tlist])
I2 = np.trapezoid(ft**2, tlist)
alpha_tot = np.pi / (2.0 * I2)
QD.register_flying_mode(
    FLY_ENV_1,
    label="first",
    gaussian=gaussian,
    tpe_alpha_X1=alpha_tot / 2,
    tpe_alpha_X2=alpha_tot / 2,
)

plotter = QDPlotter(QD, tlist=tlist)

op1 = Operation(
    CompositeOperationType.Expression,
    expr=("s_mult", 1, "a_dag"),
    context={
        "a_dag": lambda d: jnp.array(
            rotated_ladder_operator(
                dim=d[0],
                theta=QD.THETA,
                phi=QD.PHI,
                pol=Pol.PLUS,
                operator=Ladder.A_DAG,
                normalized=True,
            )
        ),
    },
    state_types=[Fock, Fock],
)

op2 = Operation(
    CompositeOperationType.Expression,
    expr=("s_mult", 1, "a_dag"),
    context={
        "a_dag": lambda d: jnp.array(
            rotated_ladder_operator(
                dim=d[0],
                theta=QD.THETA,
                phi=QD.PHI,
                pol=Pol.MINUS,
                operator=Ladder.A_DAG,
                normalized=True,
            )
        ),
    },
    state_types=[Fock, Fock],
)
print(QD.modes)

plotter.apply_operation(op1, "first_TPE")
plotter.apply_operation(op2, "first_TPE")

fig = plot_qd_comparison(
    [plotter],
    titles=[""],
    show_top=True,
    filename="multiplot_tpe",
    figsize=(4.6, 2.5),
)
