from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smef.core.units import QuantityLike, as_quantity, magnitude


@dataclass(frozen=True)
class FieldAmplitude:
    """
    Field amplitude scale for a classical drive.

    E0 is the peak electric field magnitude in V/m.
    The envelope provides the dimensionless time shape.
    """

    E0: QuantityLike
    label: Optional[str] = None

    _E0_V_m: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        E0q = as_quantity(self.E0, "V/m")
        object.__setattr__(self, "E0", E0q)
        object.__setattr__(self, "_E0_V_m", float(magnitude(E0q, "V/m")))

    def E0_V_m(self) -> float:
        return float(self._E0_V_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "field_amplitude",
            "label": self.label,
            "E0": {"value": float(magnitude(self.E0, "V/m")), "unit": "V/m"},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldAmplitude":
        E = data["E0"]
        return cls(
            E0=as_quantity(float(E["value"]), str(E.get("unit", "V/m"))),
            label=data.get("label"),
        )
