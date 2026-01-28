from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import math
import numpy as np

from smef.core.units import QuantityLike, as_quantity, magnitude

from .base import SerializableEnvelopeU


@dataclass(frozen=True)
class SymbolicEnvelopeU(SerializableEnvelopeU):
    """
    Unitful symbolic envelope evaluated via restricted eval.

    Expression is evaluated with locals:
      - t: time in `t_unit` as float
      - np: numpy
      - math: math
      - params: merged directly into locals (dimensionless floats)

    Safety:
      - globals uses {"__builtins__": {}} only
    """

    expr: str
    params: Mapping[str, float]
    t_unit: str = "s"

    def __post_init__(self) -> None:
        if not isinstance(self.expr, str) or not self.expr.strip():
            raise ValueError("expr must be a non-empty string")
        if not isinstance(self.t_unit, str) or not self.t_unit.strip():
            raise ValueError("t_unit must be a non-empty string")

        # Normalize params to plain dict[str, float]
        p: Dict[str, float] = {}
        for k, v in dict(self.params).items():
            p[str(k)] = float(v)
        object.__setattr__(self, "params", p)

        # Validate unit string by attempting a conversion on a dummy time quantity.
        # This will raise if t_unit is unknown.
        _ = as_quantity(0.0, "s").to(self.t_unit)

    def __call__(self, t: QuantityLike) -> float:
        if not hasattr(t, "to"):
            raise TypeError("EnvelopeU requires unitful time (QuantityLike).")

        t_val = float(magnitude(t, self.t_unit))
        local: Dict[str, Any] = {"t": t_val, "np": np, "math": math}
        local.update(self.params)

        # Restricted eval (no builtins).
        return float(eval(self.expr, {"__builtins__": {}}, local))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "symbolic",
            "expr": self.expr,
            "params": dict(self.params),
            "t_unit": self.t_unit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolicEnvelopeU":
        expr = str(data["expr"])
        t_unit = str(data.get("t_unit", "s"))
        params_in = dict(data.get("params", {}))
        params = {str(k): float(v) for k, v in params_in.items()}
        return cls(expr=expr, params=params, t_unit=t_unit)
