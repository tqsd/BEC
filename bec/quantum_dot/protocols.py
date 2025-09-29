from typing import Any, Dict, List, Literal, Protocol, Tuple, Union
from qutip import Qobj
from bec.light.light_mode import LightMode
from bec.params.transitions import Transition, TransitionType


class ContextProvider(Protocol):
    def build(self) -> Dict[str, Any]: ...


class ModeProvider(Protocol):
    """
    Protocol for objects that expose a collection of light modes and basic
    registry/query operations. Implementations typically combine intrinsic
    (built-in) and external (runtime) modes.
    """

    @property
    def modes(self) -> List[LightMode]:
        """
        All available modes exposed by the provider, typically ordered as
        intrinsic first, then external.
        """
        ...

    @property
    def intrinsic(self) -> List[LightMode]:
        """
        All intrinsic available modes exposed by the provider
        """
        ...

    @property
    def external(self) -> List[LightMode]:
        """
        All external available modes exposed by the provider
        """
        ...

    def register_external(self, *args, **kwargs) -> None:
        """
        Register an external mode at runtime.

        Implementations may accept a `LightMode` instance or arguments
        sufficient to construct one.
        """
        ...

    def by_transition_and_source(
        self, transition: Transition, source: TransitionType
    ) -> Tuple[int, LightMode]:
        """
        Look up the first mode matching a transition and source.

        Returns
        -------
        (int, LightMode)
            Index in `modes` and the corresponding mode.

        Raises
        ------
        ValueError
            If no matching mode exists.
        """
        ...

    def reset(self) -> None:
        """
        Remove all externally registered modes while keeping intrinsic ones.
        """
        ...


class HamiltonianProvider(Protocol):
    def fss(self, dims: List[int], time_unit_s: float) -> Qobj: ...
    def classical_2g_flip(self, dims: List[int]) -> Qobj: ...
    def classical_2g_detuning(self, dims: List[int]) -> Qobj: ...


class CollapseProvider(Protocol):
    def qutip_collapse_ops(self, dims: List[int]) -> List[Qobj]: ...


class ObservableProvider(Protocol):
    def qd_projectors(self, dims: List[int]) -> Dict[str, Qobj]: ...
    def light_mode_projectors(
        self, dims: List[int], include_qd: bool
    ) -> Dict[str, Qobj]: ...


class DiagnosticsProvider(Protocol):
    def effective_overlap(
        self, which: Literal["early", "late", "avg"]
    ) -> float: ...

    def mode_layout_summary(self) -> Dict[str, Any]: ...
