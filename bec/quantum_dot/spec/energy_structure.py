from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smef.core.units import QuantityLike

from bec.core.units import as_eV

EnergyLike = Any


@dataclass(frozen=True)
class EnergyStructure:
    """
    Four-level QD energies (absolute, in eV):
      G = 0
      X1, X2 (split by FSS around exciton center)
      XX (biexciton), optionally defined by binding energy
    """

    X1: QuantityLike
    X2: QuantityLike
    XX: QuantityLike
    G: QuantityLike = as_eV(0.0)

    @classmethod
    def from_levels(
        cls,
        *,
        X1: EnergyLike,
        X2: EnergyLike,
        XX: EnergyLike,
    ) -> EnergyStructure:
        obj = cls(X1=as_eV(X1), X2=as_eV(X2), XX=as_eV(XX))
        obj.validate()
        return obj

    @classmethod
    def from_params(
        cls,
        *,
        exciton: EnergyLike,
        fss: EnergyLike = 0.0,
        binding: EnergyLike = 0.0,
    ) -> EnergyStructure:
        EX = as_eV(exciton).to("eV")
        d = as_eV(fss).to("eV")
        Eb = as_eV(binding).to("eV")

        X1 = (EX + 0.5 * d).to("eV")
        X2 = (EX - 0.5 * d).to("eV")
        XX = (2.0 * EX - Eb).to("eV")

        obj = cls(X1=X1, X2=X2, XX=XX)
        obj.validate()
        return obj

    @property
    def exciton_center(self) -> QuantityLike:
        return (0.5 * (self.X1.to("eV") + self.X2.to("eV"))).to("eV")

    @property
    def fss(self) -> QuantityLike:
        return (self.X1.to("eV") - self.X2.to("eV")).to("eV")

    @property
    def binding(self) -> QuantityLike:
        return (2.0 * self.exciton_center.to("eV") - self.XX.to("eV")).to("eV")

    def energies_eV(self) -> dict[str, float]:
        return {
            "G": float(self.G.to("eV").magnitude),
            "X1": float(self.X1.to("eV").magnitude),
            "X2": float(self.X2.to("eV").magnitude),
            "XX": float(self.XX.to("eV").magnitude),
        }

    def validate(self) -> None:
        x1 = float(self.X1.to("eV").magnitude)
        x2 = float(self.X2.to("eV").magnitude)
        xx = float(self.XX.to("eV").magnitude)

        if x1 <= 0.0 or x2 <= 0.0 or xx <= 0.0:
            raise ValueError(
                f"Energies must be > 0 eV: X1={x1}, X2={x2}, XX={xx}"
            )

        if xx <= max(x1, x2):
            raise ValueError(f"Expected XX > X1,X2: XX={xx}, X1={x1}, X2={x2}")
