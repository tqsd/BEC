from __future__ import annotations

from typing import Any, Protocol, TypeAlias, cast
import pint
from pint import DimensionalityError
import numpy as np

ureg = pint.UnitRegistry()

# project aliases (do once, globally)
ureg.define("ps = picosecond")
ureg.define("um = micrometer")
# ureg.define("GHz = gigahertz")  # usually already present
ureg.define("eV = 1.602176634e-19 joule")


class QuantityLike(Protocol):
    @property
    def magnitude(self) -> Any: ...
    @property
    def units(self) -> Any: ...
    def to_base_units(self) -> QuantityLike: ...
    def to(self, unit: str) -> QuantityLike: ...
    def __add__(self, other: Any) -> QuantityLike: ...
    def __radd__(self, other: Any) -> QuantityLike: ...
    def __sub__(self, other: Any) -> QuantityLike: ...
    def __rsub__(self, other: Any) -> QuantityLike: ...
    def __mul__(self, other: Any) -> QuantityLike: ...
    def __rmul__(self, other: Any) -> QuantityLike: ...
    def __truediv__(self, other: Any) -> QuantityLike: ...
    def __rtruediv__(self, other: Any) -> QuantityLike: ...
    def __pow__(self, other: Any) -> QuantityLike: ...


def Q(value: Any, units: str) -> QuantityLike:
    """Create a quantity (or cast existing) in a single registry."""
    return cast(QuantityLike, ureg.Quantity(value, units))


def as_quantity(x: Any, units: str) -> QuantityLike:
    """
    Coerce x to a pint quantity with given units and verify compatibility.
    - bare numbers: interpreted as `units`
    - pint quantities: converted to `units` (raises if incompatible)
    """
    q = x if hasattr(x, "to") else Q(float(x), units)
    try:
        return cast(QuantityLike, q.to(units))
    except DimensionalityError as e:
        raise TypeError(
            f"Incompatible units: got {getattr(
                q, 'units', None)}, expected {units}"
        ) from e


def magnitude(x: Any, units: str) -> float:
    """Return float magnitude in requested units (with compatibility check)."""
    q = as_quantity(x, units)
    return float(q.to(units).magnitude)


def magnitudes(x: Any, units: str) -> np.ndarray:
    """
    Vectorized magnitudes: accepts a quantity array or numeric array.
    """
    if hasattr(x, "to"):
        return np.asarray(x.to(units).magnitude, dtype=float)
    return np.asarray(x, dtype=float)


# CONSTANTS
c = Q(299_792_458, "m/s")
h = Q(6.62607015e-34, "J*s")
hbar = Q(1.054571817e-34, "J*s")
e = Q(1.602176634e-19, "C")
epsilon_0 = Q(8.8541878128e-12, "F/m")
kB = Q(1.380649e-23, "J/K")

# CONVERSIONS


def energy_to_wavelength(E: Any, *, out_unit: str = "nm") -> QuantityLike:
    """
    Convert photon energy to vacuum wavelength using λ = h c / E.

    Parameters
    ----------
    E:
        Energy-like (QuantityLike or float assumed to be in eV via as_quantity)
    out_unit:
        Output unit, typically "nm" or "m".

    Returns
    -------
    QuantityLike
        Wavelength quantity in out_unit.
    """
    E_q = as_quantity(E, "eV").to("J")
    lam = (h * c / E_q).to(out_unit)
    return lam


def wavelength_to_energy(lam: Any, *, out_unit: str = "eV") -> QuantityLike:
    """
    Convert vacuum wavelength to photon energy using E = h c / λ.

    Parameters
    ----------
    lam:
        Wavelength-like (QuantityLike or float assumed to be in nm via as_quantity)
    out_unit:
        Output unit, typically "eV" or "J".

    Returns
    -------
    QuantityLike
        Energy quantity in out_unit.
    """
    lam_q = as_quantity(lam, "nm").to("m")
    E = (h * c / lam_q).to(out_unit)
    return E


def energy_to_rad_s(E: Any, *, out_unit: str = "rad/s"):
    E_J = as_quantity(E, "eV").to("J")
    return (E_J / hbar).to(out_unit)
