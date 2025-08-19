from matplotlib.lines import Line2D
import jax.numpy as jnp
import numpy as np
from photon_weave.operation import (
    CompositeOperationType,
    Operation,
)
from photon_weave.state.fock import Fock
from photon_weave.state.envelope import Envelope
from bec.light.envelopes import GaussianEnvelope
from bec.operators.fock_operators import (
    Ladder,
    Pol,
    rotated_ladder_operator,
)
from bec.operators.qd_operators import QDState
from bec.plots.multiplot import plot_qd_comparison
from bec.plots.plotter import QDPlotter
from bec.quantum_dot.params import CavityParams, DipoleParams, EnergyLevels
from bec.quantum_dot.qd_photon_weave import QuantumDotSystem

# --- Step 1: Make energy levels & QD system ---
EL = EnergyLevels(exciton=1.35, biexciton=2.7, fss=0.1)
DP = DipoleParams(dipole_moment_Cm=3.0e-29)
CP = CavityParams(Q=1e4, Veff_um3=1.0, lambda_nm=920.0, n=3.5)
QD1 = QuantumDotSystem(
    EL, initial_state=QDState.G, cavity_params=CP, dipole_params=DP
)

QD2 = QuantumDotSystem(
    EL, initial_state=QDState.G, cavity_params=CP, dipole_params=DP
)

FLY_ENV_1 = Envelope()
FLY_ENV_1.wavelength = 951.41
FLY_ENV_2 = Envelope()
FLY_ENV_2.wavelength = 883.46

QD1.register_flying_mode(
    FLY_ENV_1,
    gaussian=GaussianEnvelope(t0=2, sigma=0.25, area=np.pi / 2),
    label="first",
)

QD2.register_flying_mode(
    FLY_ENV_1,
    gaussian=GaussianEnvelope(t0=2, sigma=0.25, area=np.pi / 2),
    label="first",
)
QD2.register_flying_mode(
    FLY_ENV_2,
    gaussian=GaussianEnvelope(t0=2, sigma=0.25, area=np.pi / 2),
    label="second",
)

# --- Step 2: Build optical modes / full composite space ---

plotter1 = QDPlotter(QD1, tlist=np.linspace(0, 10, 1000))
plotter2 = QDPlotter(QD2, tlist=np.linspace(0, 10, 1000))


op11 = Operation(
    CompositeOperationType.Expression,
    expr=("s_mult", 1, "a_dag"),
    context={
        "a_dag": lambda d: jnp.array(
            rotated_ladder_operator(
                dim=d[0],
                theta=QD1.THETA,
                phi=QD1.PHI,
                pol=Pol.MINUS,
                operator=Ladder.A_DAG,
                normalized=True,
            )
        ),
    },
    state_types=[Fock, Fock],
)

op21 = Operation(
    CompositeOperationType.Expression,
    expr=("s_mult", 1, "a_dag"),
    context={
        "a_dag": lambda d: jnp.array(
            rotated_ladder_operator(
                dim=d[0],
                theta=QD2.THETA,
                phi=QD2.PHI,
                pol=Pol.MINUS,
                operator=Ladder.A_DAG,
                normalized=True,
            )
        ),
    },
    state_types=[Fock, Fock],
)

op22 = Operation(
    CompositeOperationType.Expression,
    expr=("s_mult", 1, "a_dag"),
    context={
        "a_dag": lambda d: jnp.array(
            rotated_ladder_operator(
                dim=d[0],
                theta=QD2.THETA,
                phi=QD2.PHI,
                pol=Pol.MINUS,
                operator=Ladder.A_DAG,
                normalized=True,
            )
        ),
    },
    state_types=[Fock, Fock],
)


plotter1.apply_operation(op11, "first")
plotter2.apply_operation(op21, "first")
plotter2.apply_operation(op22, "second")

fig = plot_qd_comparison(
    [plotter1, plotter2],
    titles=["Exciton", "Biexciton"],
    show_top=True,
    filename="multiplot_interaction",
    figsize=(9.2, 2.5),
)
