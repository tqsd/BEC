from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from bec.units import as_quantity, magnitude

from .base import SerializableEnvelope, TimeLike, _time_s


@dataclass(frozen=True)
class TabulatedEnvelope(SerializableEnvelope):
    """
    Piecewise-linear envelope defined by samples.

    Times are unitful in the API, but internally stored in seconds floats.

    Attributes:
      t_s: strictly increasing time samples in seconds (floats), len >= 2
      y: corresponding values, same length as t_s
    """

    t_s: Tuple[float, ...]
    y: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.t_s) != len(self.y) or len(self.t_s) < 2:
            raise ValueError("t and y must have the same length >= 2")
        if any(np.isnan(self.t_s)) or any(np.isnan(self.y)):
            raise ValueError("NaNs in t or y are not allowed")
        if any(
            self.t_s[i] >= self.t_s[i + 1] for i in range(len(self.t_s) - 1)
        ):
            raise ValueError("t must be strictly increasing")

    @classmethod
    def from_samples(
        cls, t: Iterable[Any], y: Iterable[Any]
    ) -> "TabulatedEnvelope":
        t_s = tuple(float(magnitude(as_quantity(v, "s"), "s")) for v in t)
        y_f = tuple(float(v) for v in y)
        return cls(t_s=t_s, y=y_f)

    def __call__(self, t: TimeLike) -> float:
        t_s = _time_s(t)
        return float(
            np.interp(t_s, self.t_s, self.y, left=self.y[0], right=self.y[-1])
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tabulated",
            "t": {"values": [float(v) for v in self.t_s], "unit": "s"},
            "y": [float(v) for v in self.y],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabulatedEnvelope":
        t_blk = data["t"]
        unit = str(t_blk.get("unit", "s"))
        t_vals = [as_quantity(float(v), unit) for v in t_blk["values"]]
        t_s = tuple(float(magnitude(v, "s")) for v in t_vals)
        y = tuple(float(v) for v in data["y"])
        return cls(t_s=t_s, y=y)
