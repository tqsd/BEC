from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np

from smef.core.units import QuantityLike


@runtime_checkable
class EnvelopeU(Protocol):
    """
    Unitful envelope.

    - Parameters are stored as quantities (e.g. ps).
    - __call__ accepts a time quantity.
    - returns a dimensionless float.
    """

    def __call__(self, t: QuantityLike) -> float: ...


@runtime_checkable
class SerializableEnvelopeU(EnvelopeU, Protocol):
    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializableEnvelopeU": ...


@runtime_checkable
class CompiledEnvelope(Protocol):
    """
    Unitless callable compatible with SMEF coefficient evaluation.

    The convention is:
      t_phys_s = t_solver * time_unit_s
    """

    def eval(self, t_solver: np.ndarray, time_unit_s: float) -> np.ndarray: ...


@dataclass(frozen=True)
class TimeBasisU:
    """
    Optional helper for envelopes parameterized in some time unit.

    Example:
      basis = TimeBasisU(Q(1, "ps"))
      x = basis.to_dimless(Q(3, "ps"))  # 3.0
    """

    unit: QuantityLike

    def __post_init__(self) -> None:
        u = self.unit
        if not hasattr(u, "to"):
            raise TypeError("TimeBasisU.unit must be a quantity")
        # store as seconds internally
        object.__setattr__(self, "unit", u.to("s"))
        if float(self.unit.magnitude) <= 0.0:
            raise ValueError("TimeBasisU.unit must be > 0")

    def to_dimless(self, t: QuantityLike) -> float:
        return float((t.to("s") / self.unit).magnitude)

    def to_phys(self, x: float) -> QuantityLike:
        return self.unit * float(x)
