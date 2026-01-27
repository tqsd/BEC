from dataclasses import dataclass
from typing import Literal, Union
from bec.units import QuantityLike, as_quantity


@dataclass(frozen=True)
class FieldAmplitude:
    E0: QuantityLike = None  # V/m

    def __post_init__(self):
        object.__setattr__(self, "E0", as_quantity(self.E0, "V/m"))
