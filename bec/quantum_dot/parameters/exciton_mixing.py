from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from bec.quantum_dot.units import as_eV
from bec.units import QuantityLike, as_quantity


@dataclass(frozen=True)
class ExcitonMixingParams:
    """
    Exciton subspace anisotropic mixing parameter.

    delta_prime:
      Off-diagonal coupling δ' in eV in the {X1, X2} basis.
      (Real for now; can be extended to complex later.)
    """

    delta_prime: QuantityLike = as_eV(0.0)

    @classmethod
    def from_values(cls, *, delta_prime_eV: Any = 0.0) -> "ExcitonMixingParams":
        return cls(delta_prime=as_eV(delta_prime_eV))
