import jax.numpy as jnp
from typing import Callable, Dict, List
from photon_weave.extra import interpreter
from qutip import Qobj
from bec.operators.qd_operators import QDState, transition_operator
from bec.operators.fock_operators import (
    rotated_ladder_operator,
    Ladder,
    Pol,
    vacuum_projector,
)
from bec.quantum_dot.protocols import ContextProvider, ModeProvider


class QDContextBuilder(ContextProvider):
    def __init__(self, mode_provider: ModeProvider, theta: float, phi: float):
        self._modes = mode_provider
        self._theta = float(theta)
        self._phi = float(phi)

    def build(self) -> Dict[str, Callable]:
        ctx: Dict[str, Callable] = {
            # QD ops
            "s_XX_G": lambda _: jnp.array(
                transition_operator(QDState.XX, QDState.G)
            ),
            "s_XX_X1": lambda _: jnp.array(
                transition_operator(QDState.XX, QDState.X1)
            ),
            "s_XX_X2": lambda _: jnp.array(
                transition_operator(QDState.XX, QDState.X2)
            ),
            "s_X1_G": lambda _: jnp.array(
                transition_operator(QDState.X1, QDState.G)
            ),
            "s_X2_G": lambda _: jnp.array(
                transition_operator(QDState.X2, QDState.G)
            ),
            "s_G_X1": lambda _: jnp.array(
                transition_operator(QDState.G, QDState.X1)
            ),
            "s_G_X2": lambda _: jnp.array(
                transition_operator(QDState.G, QDState.X2)
            ),
            "s_G_XX": lambda _: jnp.array(
                transition_operator(QDState.G, QDState.XX)
            ),
            "s_X1_XX": lambda _: jnp.array(
                transition_operator(QDState.X1, QDState.XX)
            ),
            "s_X2_XX": lambda _: jnp.array(
                transition_operator(QDState.X2, QDState.XX)
            ),
            "s_X1_X1": lambda _: jnp.array(
                transition_operator(QDState.X1, QDState.X1)
            ),
            "s_X1_X2": lambda _: jnp.array(
                transition_operator(QDState.X1, QDState.X2)
            ),
            "s_X2_X1": lambda _: jnp.array(
                transition_operator(QDState.X2, QDState.X1)
            ),
            "s_X2_X2": lambda _: jnp.array(
                transition_operator(QDState.X2, QDState.X2)
            ),
            "s_XX_XX": lambda _: jnp.array(
                transition_operator(QDState.XX, QDState.XX)
            ),
            "idq": lambda _: jnp.eye(4),
        }

        THETA, PHI = self._theta, self._phi
        for i, _m in enumerate(self._modes.modes):
            s = 1 + i * 2
            ctx.update(
                {
                    f"a{i}+": lambda d, _s=s: rotated_ladder_operator(
                        d[_s], THETA, PHI, operator=Ladder.A, pol=Pol.PLUS
                    ),
                    f"a{i}+_dag": lambda d, _s=s: rotated_ladder_operator(
                        d[_s], THETA, PHI, operator=Ladder.A_DAG, pol=Pol.PLUS
                    ),
                    f"a{i}-": lambda d, _s=s: rotated_ladder_operator(
                        d[_s], THETA, PHI, operator=Ladder.A, pol=Pol.MINUS
                    ),
                    f"a{i}-_dag": lambda d, _s=s: rotated_ladder_operator(
                        d[_s], THETA, PHI, operator=Ladder.A_DAG, pol=Pol.MINUS
                    ),
                    f"n{i}+": lambda d, _s=s: rotated_ladder_operator(
                        d[_s], THETA, PHI, pol=Pol.PLUS, operator=Ladder.A_DAG
                    )
                    @ rotated_ladder_operator(
                        d[_s], THETA, PHI, pol=Pol.PLUS, operator=Ladder.A
                    ),
                    f"n{i}-": lambda d, _s=s: rotated_ladder_operator(
                        d[_s], THETA, PHI, pol=Pol.MINUS, operator=Ladder.A_DAG
                    )
                    @ rotated_ladder_operator(
                        d[_s], THETA, PHI, pol=Pol.MINUS, operator=Ladder.A
                    ),
                    f"if{i}": lambda d, _s=s: jnp.eye(d[_s] * d[_s + 1]),
                    f"vac{i}": lambda d, _s=s: vacuum_projector(d[_s]),
                }
            )
        return ctx
