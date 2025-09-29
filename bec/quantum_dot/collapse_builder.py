from typing import Dict, Any, List, Tuple

from qutip import Qobj
import numpy as np
import jax.numpy as jnp
from bec.light.light_mode import LightMode
from bec.params.transitions import Transition, TransitionType
from bec.quantum_dot.kron_pad_utility import KronPad
from bec.quantum_dot.protocols import CollapseProvider, ModeProvider
from photon_weave.extra import interpreter


class CollapseBuilder(CollapseProvider):
    """
    Builder for collapse (Lindblad) operators in quantum-dot simulations

    This class uses decay rates (`gammas`), a symbolic operator context,
    a KronPad utility, and a ModeProvider to construct the collapse
    operators that describe spontaneous emission from quantum-dot into
    photonic modes. The resulting operators are returend as QuTiP `Qobj`
    matrices and are passed directly to solvers.

    The builder supports two configurations:
    - Two intrinsic modes (degenerate exciton case):
      Uses a single internal mode for the XX->X transition and a single
      internal mode for the X->G transition.
    - Four intrinsic modes (splic exciton case):
      Uses separate internal modes for XX->X1, XX->X2, X1->G, X2->G

    Parameters
    ----------
    gammas: dict[str, float]
        Mapping of symbolic decay labels (e.g. "L_XX_X1", "L_X1_G")
        to their rates. Each rate is scaled by `sqrt(time_unit_s)` when
        constructing operators.
    context: dict[str, Any]
        Context dictionary mapping cymbolic operator labels to callables.
        Passed through the `interpreter`
    kron: KronPad
        Utility for padding local operators into the full Hilbert space
    mode_provider: ModeProvider
        Provider access to photonic modes (with source and transition
        attributes). Determines which modes are used for each collapse
        channel

    Notes
    -----
    - The CollapseBuilder is used in the `QuantumDot` class and is not
      meant to be used on its own.
    - All collapse operators are returned as sparse `Qobj` in CSR format.
    - The symbolic operator expressions are evaluated by
      `photon_weave.extra.interpreter`, which must understand the keys
      produced by `KronPad` and the quantum-dot context.

    """

    def __init__(
        self,
        gammas: Dict[str, float],
        context: Dict[str, Any],
        kron: KronPad,
        mode_provider: ModeProvider,
    ):
        self._g = gammas
        self._ctx = context
        self._kron = kron
        self._modes = mode_provider

    def by_transition_and_source(
        self, transition: Transition, source: TransitionType
    ) -> Tuple[int, LightMode]:
        """
        Find the first mode matching a given transition and source.

        Parameters
        ----------
        transition : Transition
            The physical transition label (e.g., Transition.G_X1).
        source : TransitionType
            The origin of the mode (e.g., INTERNAL, EXTERNAL).

        Returns
        -------
        (int, LightMode)
            Tuple of (index in `modes`, the matching LightMode).

        Raises
        ------
        ValueError
            If no matching mode is found.
        """
        for i, m in enumerate(self._modes.modes):
            if (
                m.source == source
                and getattr(m, "transition", None) == transition
            ):
                return i, m
        raise ValueError(
            f"No mode with transition {transition} and source {source}"
        )

    def qutip_collapse_ops(
        self, dims: List[int], time_unit_s: float = 1.0
    ) -> List[Qobj]:
        """
        Build QuTiP collapse operators for the configured quantum-dot
        to model decays and return them as Qobj matrices.

        This function assembles symbolic operatr expressions that model
        spontaneous-emission channels. It chooses between two layouts:

        - If there are two intrinsic modes (degenerate exciton levels),
          it expects a single internal mode for XX->X and a single
          internal mode fox X->G, and builds two "add" expressions
          (each the sum of two scaled, padded ladder operators).
        - Otherwise it expects separate internal modes per branch (X1_XX,
          X2_XX, G_X1, G_X2), and builds the two "add" expressions
          accordingly.

        Each symbolic expression is passed to `photon_weave.extra.interpreter`
        with the context and dimensions. The interpreter returns a square
        matrix (`jax.numpy.ndarray`) with dimensions equal to the product
        of `dims`. Then we wrap the result as a QuTiP Qobj in CSR (compressed
        sparse row) format with `dims=[dims, dims]`.

        Parameters
        ----------
        dims: List[int]
            Composite Hilbert-Space dimension in the same order used by the
            interpreter and KronPad
        time_unit_s: float, optional
            Time scaling in seconds. Decay rates in `self._g` are scaled by
            `sqrt(time_unit_s)` so that the collapse operators have the same
            units for the chosen integration step.

        Returns
        -------
        list[qupit.Qobj]
            A list of collapse operators as QuTip Qobj matrices

        Raises
        ------
        ValueError
            If a required mode cannot be found in the provided ModeProvider.
        """
        s = float(time_unit_s)
        if len(self._modes.intrinsic) == 2:
            i_xxx = self._modes.by_transition_and_source(
                Transition.X_XX, TransitionType.INTERNAL
            )[0]
            i_xg = self._modes.by_transition_and_source(
                Transition.G_X, TransitionType.INTERNAL
            )[0]
            ops = (
                (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self._g["L_XX_X1"] * s),
                        self._kron.pad("s_XX_X1", "a+_dag", i_xxx),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self._g["L_XX_X2"] * s),
                        self._kron.pad("s_XX_X2", "a-_dag", i_xxx),
                    ),
                ),
                (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self._g["L_X1_G"] * s),
                        self._kron.pad("s_X1_G", "a-_dag", i_xg),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self._g["L_X2_G"] * s),
                        self._kron.pad("s_X2_G", "a+_dag", i_xg),
                    ),
                ),
            )
        else:
            i_xxx1 = self._modes.by_transition_and_source(
                Transition.X1_XX, TransitionType.INTERNAL
            )[0]
            i_xxx2 = self._modes.by_transition_and_source(
                Transition.X2_XX, TransitionType.INTERNAL
            )[0]
            i_x1g = self._modes.by_transition_and_source(
                Transition.G_X1, TransitionType.INTERNAL
            )[0]
            i_x2g = self._modes.by_transition_and_source(
                Transition.G_X2, TransitionType.INTERNAL
            )[0]
            ops = (
                (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self._g["L_XX_X1"] * s),
                        self._kron.pad("s_XX_X1", "a+_dag", i_xxx1),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self._g["L_XX_X2"] * s),
                        self._kron.pad("s_XX_X2", "a-_dag", i_xxx2),
                    ),
                ),
                (
                    "add",
                    (
                        "s_mult",
                        jnp.sqrt(self._g["L_X1_G"] * s),
                        self._kron.pad("s_X1_G", "a-_dag", i_x1g),
                    ),
                    (
                        "s_mult",
                        jnp.sqrt(self._g["L_X2_G"] * s),
                        self._kron.pad("s_X2_G", "a+_dag", i_x2g),
                    ),
                ),
            )

        out = []
        for op in ops:
            arr = interpreter(op, self._ctx, dims)
            out.append(Qobj(np.array(arr), dims=[dims, dims]).to("csr"))
        return out
