from __future__ import annotations

from dataclasses import dataclass

from bec.units import QuantityLike, as_quantity


def _as_um3(x) -> QuantityLike:
    return as_quantity(x, "um**3")


def _as_nm(x) -> QuantityLike:
    return as_quantity(x, "nm")


def _as_dimless(x) -> float:
    # dimensionless: accept only scalars
    return float(x)


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
        - float interpreted as µm³
    lambda_cav : QuantityLike
        Cavity wavelength.
        - float interpreted as nm
    n : float, default=3.5
        Refractive index (dimensionless).

    Notes
    -----
    Used for Purcell factor:

    .. math::
        F_p = \frac{3}{4\pi^2}\Big(\frac{\lambda}{n}\Big)^3 \frac{Q}{V_{eff}}
    """

    Q: float
    Veff: QuantityLike
    lambda_cav: QuantityLike
    n: float = 3.5

    # ---------- constructors ----------

    @classmethod
    def from_values(
        cls,
        *,
        Q: float,
        Veff_um3,
        lambda_nm,
        n: float = 3.5,
    ) -> "CavityParams":
        return cls(
            Q=_as_dimless(Q),
            Veff=_as_um3(Veff_um3),
            lambda_cav=_as_nm(lambda_nm),
            n=_as_dimless(n),
        )

    # ---------- normalized / derived ----------

    @property
    def Veff_m3(self) -> QuantityLike:
        """Effective mode volume in m³."""
        return self.Veff.to("m**3")

    @property
    def lambda_m(self) -> QuantityLike:
        """Cavity wavelength in meters."""
        return self.lambda_cav.to("m")

    def validate(self) -> None:
        if self.Q <= 0:
            raise ValueError(f"Cavity Q must be > 0, got {self.Q}")
        if self.n <= 0:
            raise ValueError(f"Refractive index n must be > 0, got {self.n}")

        V = float(self.Veff_m3.magnitude)
        lam = float(self.lambda_m.magnitude)

        if V <= 0:
            raise ValueError(f"Veff must be > 0, got {self.Veff}")
        if lam <= 0:
            raise ValueError(f"lambda must be > 0, got {self.lambda_cav}")

    # ---------- numeric view ----------

    def as_floats(self) -> dict[str, float]:
        """
        Float-only view (SI units) for fast numeric code.
        """
        return {
            "Q": self.Q,
            "Veff_m3": float(self.Veff_m3.magnitude),
            "lambda_m": float(self.lambda_m.magnitude),
            "n": self.n,
        }
