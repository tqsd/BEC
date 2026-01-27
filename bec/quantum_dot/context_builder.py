import jax.numpy as jnp
from typing import Callable, Dict
from bec.operators.qd_operators import (
    QDState,
    basis_ket_4d,
    exciton_eigenkets_4d,
    outer_operator,
    transition_operator,
)
from bec.operators.fock_operators import (
    rotated_ladder_operator,
    Ladder,
    Pol,
    vacuum_projector,
)
from bec.params.energy_levels import EnergyLevels
from bec.quantum_dot.protocols import ContextProvider, ModeProvider


class QDContextBuilder(ContextProvider):
    """
    Context builder for quantum-dot simulations

    This class creates a mapping from string keys to callables (the `context`)
    that the `photon_weave.extra.interpreter` can use to resolve symbolic
    operator expressions into actual matrices.

    The context includes:
    - Intrinsic quantum-dot operators (transition operators between biexciton,
      exciton, and ground state).
    - Identity operators and projectors for the quantum-dot subspace.
    - Rotated ladder operators for each photonic mode provided by
      `ModeProvider`. These account for polarization angles (`theta`, `phi`)
      and are indexed by mode number.
    - Number operators (`n{i}+`, `n{i}-`) built as products of rotated ladder
    operatrs for each polarization.
    - Auxiliary operators such as per-mode identity (`if{i}`) and vacuum
      projectors (`vac{i}`).

    Parameters
    ----------
    mode_provider : ModeProdiver
        Provider of photonic modes. Determines how many ladder operator sets
        will be included in the context.
    theta: float
        Polarizatio rotation angle (in radians) used for rotated ladder
        operatrs.
    phi: float
        Additional phase angle (in radians) used for rotated ladder operators.

    Examples
    --------
    >>> builder = QDContextBuilder(mode_provider, theat=0.0, phi=0.0)
    >>> ctx = builder.build()
    """

    def __init__(
        self,
        mode_provider: ModeProvider,
        energy_levels: EnergyLevels,
        theta: float,
        phi: float,
    ):
        self._modes = mode_provider
        self._theta = float(theta)
        self._phi = float(phi)
        self._EL = energy_levels

    def build(self) -> Dict[str, Callable]:
        """
        Construct the context mapping from operators names to callables

        Returns
        -------
        Dict[str, Callable]
           A dictionary where each key is a symbolic operator label and
           each value is a function
           `f(dimensions:List[int])->jax.numpy.ndarray` that returns the
           corresponding operator matrix.

        Notes
        -----
        - Quantum-dot operators are always 4x4 matrices (fixed Hilbert space).
        - Ladder operator depend on the mode dimensions provided to the
          callable at evaluation time (usually the dimension is 2)
        - Number operators are implemented as products of creation and
          annihilation operators.
        - Each mode contributes a set of operators indexeb by its position in
          the mode provider.
        """
        ctx: Dict[str, Callable] = {
            # QD ops
            "s_G_G": lambda _: jnp.array(
                transition_operator(QDState.G, QDState.G)
            ),
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
            "idq": lambda _: jnp.eye(4, dtype=jnp.complex128),
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
                    f"if{i}": lambda d, _s=s: jnp.eye(
                        d[_s] * d[_s + 1], dtype=jnp.complex128
                    ),
                    f"vac{i}": lambda d, _s=s: vacuum_projector(d[_s]),
                }
            )

        Hx = self._EL.exciton_hamiltonian_2x2()
        ket_low, ket_high = exciton_eigenkets_4d(Hx)

        S_Xm_Xp = outer_operator(ket_low, ket_high)
        S_Xp_Xm = outer_operator(ket_high, ket_low)
        P_Xm = outer_operator(ket_low)
        P_Xp = outer_operator(ket_high)

        ctx.update(
            {
                "s_Xm_Xp": lambda _d, A=S_Xm_Xp: jnp.array(A),
                "s_Xp_Xm": lambda _d, A=S_Xp_Xm: jnp.array(A),
                "s_Xm_Xm": lambda _d, A=P_Xm: jnp.array(A),
                "s_Xp_Xp": lambda _d, A=P_Xp: jnp.array(A),
            }
        )

        ket_G = basis_ket_4d(QDState.G)
        ket_XX = basis_ket_4d(QDState.XX)

        S_Xm_G = outer_operator(ket_low, ket_G)
        S_Xp_G = outer_operator(ket_high, ket_G)
        S_XX_Xm = outer_operator(ket_XX, ket_low)
        S_XX_Xp = outer_operator(ket_XX, ket_high)

        ctx.update(
            {
                "s_Xm_G": lambda _d, A=S_Xm_G: jnp.array(A),
                "s_Xp_G": lambda _d, A=S_Xp_G: jnp.array(A),
                "s_XX_Xm": lambda _d, A=S_XX_Xm: jnp.array(A),
                "s_XX_Xp": lambda _d, A=S_XX_Xp: jnp.array(A),
            }
        )
        return ctx
