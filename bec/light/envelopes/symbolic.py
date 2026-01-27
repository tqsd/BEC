from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import math

from .base import SerializableEnvelope, TimeLike, _time_s


@dataclass(frozen=True)
class SymbolicEnvelope(SerializableEnvelope):
    """
    Symbolic envelope evaluated via restricted eval.

    Expression is evaluated with:
      - t: time in seconds (float)
      - np: numpy
      - math: math
      - params: merged directly into locals

    Safety:
      - globals uses {"__builtins__": {}} only
    """

    expr: str
    params: Dict[str, float]

    def __call__(self, t: TimeLike) -> float:
        t_s = _time_s(t)
        local = {"t": float(t_s), "np": np, "math": math}
        local.update(self.params)
        return float(eval(self.expr, {"__builtins__": {}}, local))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "symbolic",
            "expr": self.expr,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolicEnvelope":
        expr = str(data["expr"])
        params = {
            str(k): float(v) for k, v in dict(data.get("params", {})).items()
        }
        return cls(expr=expr, params=params)
