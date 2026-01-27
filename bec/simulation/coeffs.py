from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, runtime_checkable

import numpy as np


Args = Mapping[str, Any]


@runtime_checkable
class Coeff(Protocol):
    """Time-dependent scalar coefficient in *solver time* units."""

    def __call__(self, t: float, args: Args) -> complex: ...


@dataclass(frozen=True)
class ConstCoeff:
    """Truly constant coefficient. Not callable by design."""

    value: complex

    def as_complex(self) -> complex:
        return complex(self.value)


@dataclass(frozen=True)
class CallableCoeff:
    """Callable coefficient (time-dependent, possibly args-dependent)."""

    fn: Coeff

    def __call__(self, t: float, args: Args) -> complex:
        return complex(self.fn(t, args))


CoeffExpr = ConstCoeff | CallableCoeff


def is_const(c: CoeffExpr | None) -> bool:
    return isinstance(c, ConstCoeff)


def eval_coeff(c: CoeffExpr, t: float, args: Args) -> complex:
    """Uniform evaluation when you *do* need a number (e.g. debugging)."""
    if isinstance(c, ConstCoeff):
        return c.as_complex()
    return c(t, args)


def fold_const_into_op(op: np.ndarray, c: ConstCoeff) -> np.ndarray:
    """Used at adapter/export boundary if you want to remove constants from coeff channel."""
    return np.asarray(op, dtype=complex) * c.as_complex()
