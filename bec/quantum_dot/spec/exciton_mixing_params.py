from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smef.core.units import QuantityLike

from bec.core.units import as_eV


@dataclass(frozen=True)
class ExcitonMixingParams:
    """
    Exciton subspace anisotropic mixing parameter.

    delta_prime:
      Off-diagonal coupling delta_prime in eV in the {X1, X2} basis.
      (Real for now; can be extended to complex later.)
    """

    delta_prime: QuantityLike = as_eV(0.0)

    @classmethod
    def from_values(cls, *, delta_prime_eV: Any = 0.0) -> ExcitonMixingParams:
        obj = cls(delta_prime=as_eV(delta_prime_eV))
        obj.validate()
        return obj

    def validate(self) -> None:
        # For now just enforce it is energy-like and finite.
        v = float(self.delta_prime.to("eV").magnitude)
        if v != v:  # NaN check without importing math
            raise ValueError("delta_prime must not be NaN")
