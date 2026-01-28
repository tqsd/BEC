from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from smef.core.units import QuantityLike

from bec.core.units import as_dimless, as_nm, as_um3


@dataclass(frozen=True)
class CavityParams:
    r"""
    Optical cavity parameters for quantum-dot simulations (unit-aware).

    Parameters
    ----------
    Q : float
        Quality factor (dimensionless).
    Veff : QuantityLike
        Effective mode volume.
        - bare numbers interpreted as um**3
    lambda_cav : QuantityLike
        Cavity wavelength.
        - bare numbers interpreted as nm
    n : float, default=3.5
        Refractive index (dimensionless).

    Notes
    -----
    Used for Purcell factor:

      F_p = (3 / (4*pi**2)) * ((lambda / n)**3) * (Q / Veff)
    """

    Q: float
    Veff: QuantityLike
    lambda_cav: QuantityLike
    n: float = 3.5

    @classmethod
    def from_values(
        cls,
        *,
        Q: Any,
        Veff_um3: Any,
        lambda_nm: Any,
        n: Any = 3.5,
    ) -> "CavityParams":
        obj = cls(
            Q=as_dimless(Q),
            Veff=as_um3(Veff_um3),
            lambda_cav=as_nm(lambda_nm),
            n=as_dimless(n),
        )
        obj.validate()
        return obj

    @property
    def Veff_m3(self) -> QuantityLike:
        return self.Veff.to("m**3")

    @property
    def lambda_m(self) -> QuantityLike:
        return self.lambda_cav.to("m")

    def validate(self) -> None:
        if float(self.Q) <= 0.0:
            raise ValueError(f"Cavity Q must be > 0, got {self.Q}")
        if float(self.n) <= 0.0:
            raise ValueError(f"Refractive index n must be > 0, got {self.n}")

        V = float(self.Veff_m3.magnitude)
        lam = float(self.lambda_m.magnitude)

        if V <= 0.0:
            raise ValueError(f"Veff must be > 0, got {self.Veff}")
        if lam <= 0.0:
            raise ValueError(f"lambda must be > 0, got {self.lambda_cav}")

    def as_floats(self) -> Dict[str, float]:
        return {
            "Q": float(self.Q),
            "Veff_m3": float(self.Veff_m3.magnitude),
            "lambda_m": float(self.lambda_m.magnitude),
            "n": float(self.n),
        }
