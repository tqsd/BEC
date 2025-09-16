from typing import List, Tuple
from bec.light.light_mode import LightMode
from bec.params.transitions import Transition, TransitionType
from bec.quantum_dot.protocols import ModeProvider


class ModeRegistry(ModeProvider):
    """
    Registry of quantum-dot light modes.

    This class holds both **intrinsic** (internally defined) and **external**
    (registered at runtime) `LightMode` instances, and exposes a unified
    read-only `modes` view. It also stores the exciton rotation parameters
    `(THETA, PHI)` that other components (e.g., context builders) may need.

    Parameters
    ----------
    intrinsic_modes : list[LightMode]
        Built-in modes derived from energy levels (e.g., G<-->X1, G<-->X2,
        X1<-->XX, X2<-->XX).
    rotation_params : (float, float)
        The exciton mixing angle and relative phase `(THETA, PHI)`.

    Notes
    -----
    - Use `register_external()` to add runtime modes (e.g., laser drives).
    - `modes` returns `intrinsic + external` in that order.
    """

    def __init__(
        self,
        intrinsic_modes: List[LightMode],
        rotation_params: Tuple[float, float],
    ):
        self._intrinsic = intrinsic_modes[:]
        self._external: List[LightMode] = []
        self.THETA, self.PHI = rotation_params

    @property
    def intrinsic(self) -> List[LightMode]:
        return self._intrinsic

    @property
    def external(self) -> List[LightMode]:
        return self._external

    @property
    def modes(self) -> List[LightMode]:
        """
        All registered modes (intrinsic first, then external).

        Returns
        -------
        list[LightMode]
            Concatenation of intrinsic and external modes.
        """
        return [*self._intrinsic, *self._external]

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
        for i, m in enumerate(self.modes):
            if (
                m.source == source
                and getattr(m, "transition", None) == transition
            ):
                return i, m
        raise ValueError(
            f"No mode with transition {transition} and source {source}"
        )

    def register_external(self, light_mode: LightMode) -> None:
        """
        Add an external (runtime) mode to the registry.

        Parameters
        ----------
        light_mode : LightMode
            The mode to register (e.g., a driving laser or detected output).
        """
        self._external.append(light_mode)

    def reset(self) -> None:
        """
        Remove all externally registered modes.

        Notes
        -----
        Intrinsic modes are preserved; only external modes are cleared.
        """
        self._external.clear()
