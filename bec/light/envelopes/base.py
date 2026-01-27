from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable, Union

from dataclasses import dataclass

from bec.units import QuantityLike, as_quantity, magnitude, magnitude


TimeLike = Union[QuantityLike, float, int]


def _time_s(t: TimeLike) -> float:
    """
    Convert time input to seconds as float.
    - QuantityLike: converted to seconds
    - numbers: interpreted as seconds
    """
    if hasattr(t, "to"):
        return magnitude(t, "s")
    return float(t)


@runtime_checkable
class Envelope(Protocol):
    """
    Callable time-dependent envelope.

    Contract:
      f(t) -> float

    Inputs:
      - t is time-like (QuantityLike preferred). If a number is passed, it is
        interpreted as seconds.

    Output:
      - dimensionless float (shape), finite for relevant t.
    """

    def __call__(self, t: TimeLike) -> float: ...


@runtime_checkable
class SerializableEnvelope(Envelope, Protocol):
    """
    Envelope that supports JSON serialization.

    Required:
      - to_dict() -> dict[str, Any]
      - from_dict(data: dict[str, Any]) -> SerializableEnvelope

    Implementations must include a "type" field in to_dict() that is used by
    the registry loader.
    """

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializableEnvelope": ...


@dataclass(frozen=True)
class TimeBasis:
    """
    Defines conversion between physical time and a dimensionless time coordinate.

    Example:
      basis = TimeBasis(Q(1, "ps"))
      t_dimless = basis.to_dimless(Q(3, "ps"))  # 3.0
      t_phys = basis.to_phys(3.0)              # 3 ps
    """

    unit: QuantityLike  # time unit, e.g. 1 ps

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", as_quantity(self.unit, "s"))
        if magnitude(self.unit, "s") <= 0.0:
            raise ValueError("TimeBasis.unit must be > 0")

    def to_dimless(self, t_phys: QuantityLike) -> float:
        return float(
            (as_quantity(t_phys, "s") / self.unit).to_base_units().magnitude
        )

    def to_phys(self, t_dimless: float) -> QuantityLike:
        return self.unit * float(t_dimless)
