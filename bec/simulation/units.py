from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pint  # type: ignore


@dataclass(frozen=True)
class TimeUnit:
    """
    Optional ergonomic wrapper.

    Runtime still stores float seconds. This helper gives you a pint.Quantity
    if pint is available.
    """

    seconds: float
    unit: str = "second"

    def as_seconds(self) -> float:
        return float(self.seconds)

    def as_pint(self) -> Any:
        if pint is None:
            raise RuntimeError("pint is not installed")
        ureg = pint.UnitRegistry()
        return self.seconds * ureg(self.unit)
