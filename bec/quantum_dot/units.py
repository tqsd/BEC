from __future__ import annotations

from typing import Any, TypeAlias
from bec.units import QuantityLike, as_quantity

EnergyLike: TypeAlias = Any


def as_eV(x: Any) -> QuantityLike:
    return as_quantity(x, "eV")


def as_rad_s(x: Any) -> QuantityLike:
    return as_quantity(x, "rad/s")


def as_1_s(x: Any) -> QuantityLike:
    return as_quantity(x, "1/s")
