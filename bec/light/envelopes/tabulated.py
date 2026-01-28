from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from smef.core.units import QuantityLike, Q, as_quantity, magnitude

from .base import SerializableEnvelopeU


def _mag(x: Any, unit: str) -> float:
    return float(magnitude(x, unit))


@dataclass(frozen=True)
class TabulatedEnvelopeU(SerializableEnvelopeU):
    """
    Piecewise-linear envelope defined by samples.

    Public API is unitful:
      - __call__ requires QuantityLike
      - from_samples accepts unitful times (or numbers interpreted as t_unit)

    Internals:
      - times stored as seconds floats for fast interpolation

    Attributes:
      t_s: strictly increasing time samples in seconds (floats), len >= 2
      y: corresponding values, same length as t_s

    Declaration:
      t_unit declares the intended unit for symbolic/serialization convenience.
      It does not change internal storage (always seconds).
    """

    t_s: Tuple[float, ...]
    y: Tuple[float, ...]
    t_unit: str = "s"

    def __post_init__(self) -> None:
        if not isinstance(self.t_unit, str) or not self.t_unit.strip():
            raise ValueError("t_unit must be a non-empty string")

        # Validate unit string
        _ = as_quantity(0.0, "s").to(self.t_unit)

        if len(self.t_s) != len(self.y) or len(self.t_s) < 2:
            raise ValueError("t and y must have the same length >= 2")

        t_arr = np.asarray(self.t_s, dtype=float)
        y_arr = np.asarray(self.y, dtype=float)

        if np.any(np.isnan(t_arr)) or np.any(np.isnan(y_arr)):
            raise ValueError("NaNs in t or y are not allowed")

        if np.any(t_arr[:-1] >= t_arr[1:]):
            raise ValueError("t must be strictly increasing")

        # Ensure tuples are floats (defensive)
        object.__setattr__(self, "t_s", tuple(float(v) for v in t_arr.tolist()))
        object.__setattr__(self, "y", tuple(float(v) for v in y_arr.tolist()))

    @classmethod
    def from_samples(
        cls,
        t: Iterable[Any],
        y: Iterable[Any],
        *,
        t_unit: str = "s",
    ) -> "TabulatedEnvelopeU":
        # Numbers are interpreted as t_unit; quantities converted to seconds.
        t_s = tuple(_mag(as_quantity(v, t_unit), "s") for v in t)
        y_f = tuple(float(v) for v in y)
        return cls(t_s=t_s, y=y_f, t_unit=t_unit)

    def __call__(self, t: QuantityLike) -> float:
        if not hasattr(t, "to"):
            raise TypeError("EnvelopeU requires unitful time (QuantityLike).")
        t_s = _mag(t, "s")
        return float(
            np.interp(t_s, self.t_s, self.y, left=self.y[0], right=self.y[-1])
        )

    def to_dict(self) -> Dict[str, Any]:
        # Store times in declared t_unit for readability, plus the unit itself.
        t_vals = [float(Q(v, "s").to(self.t_unit).magnitude) for v in self.t_s]
        return {
            "type": "tabulated",
            "t": {"values": t_vals, "unit": self.t_unit},
            "y": [float(v) for v in self.y],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabulatedEnvelopeU":
        t_blk = data["t"]
        unit = str(t_blk.get("unit", "s"))
        t_vals = [as_quantity(float(v), unit) for v in t_blk["values"]]
        t_s = tuple(_mag(v, "s") for v in t_vals)
        y = tuple(float(v) for v in data["y"])
        return cls(t_s=t_s, y=y, t_unit=unit)
