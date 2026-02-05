from __future__ import annotations

from typing import Any

from smef.core.units import QuantityLike, as_quantity


def as_dimless(x: Any) -> float:
    return float(x)


def as_eV(x: Any) -> QuantityLike:
    return as_quantity(x, "eV")


def as_s(x: Any) -> QuantityLike:
    return as_quantity(x, "s")


def as_nm(x: Any) -> QuantityLike:
    return as_quantity(x, "nm")


def as_um3(x: Any) -> QuantityLike:
    return as_quantity(x, "um**3")


def as_Cm(x: Any) -> QuantityLike:
    return as_quantity(x, "C*m")


def as_K(x: Any) -> QuantityLike:
    return as_quantity(x, "K")


def as_rate_1_s(x: Any) -> QuantityLike:
    return as_quantity(x, "1/s")


def as_s2(x: Any) -> QuantityLike:
    return as_quantity(x, "s**2")


def as_rad_s(x: Any) -> QuantityLike:
    return as_quantity(x, "rad/s")
