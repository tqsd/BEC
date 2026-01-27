from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Union

Args = Dict[str, Any]


class CoeffExpr(Protocol):
    """Uniform coefficient interface: always callable as coeff(t, args)->float."""

    def __call__(self, t: float, args: Optional[Args] = None) -> complex: ...


@dataclass(frozen=True)
class ConstCoeff:
    value: complex

    def __call__(self, t: float, args: Optional[Args] = None) -> complex:
        return complex(self.value)


@dataclass(frozen=True)
class FuncCoeff:
    fn: Callable[..., complex]
    _takes_args: bool = False

    def __post_init__(self):
        sig = inspect.signature(self.fn)
        # count required positional params
        params = list(sig.parameters.values())
        # accept (t) or (t,args)
        takes_args = len(params) >= 2
        object.__setattr__(self, "_takes_args", takes_args)

    def __call__(self, t: float, args: Optional[Args] = None) -> complex:
        args = args or {}
        if self._takes_args:
            return complex(self.fn(t, args))
        return complex(self.fn(t))


CoeffLike = Union[complex, Callable[..., complex], CoeffExpr]


def as_coeff(x: CoeffLike) -> CoeffExpr:
    """Normalize floats/functions/CoeffExpr into a CoeffExpr."""
    if isinstance(x, (ConstCoeff, FuncCoeff)):
        return x
    if callable(x):
        return FuncCoeff(x)
    return ConstCoeff(complex(x))


def scale(c: CoeffExpr, s: complex) -> CoeffExpr:
    return FuncCoeff(lambda t, args=None, cc=c, ss=s: ss * cc(t, args))


def add(c1: CoeffExpr, c2: CoeffExpr) -> CoeffExpr:
    return FuncCoeff(lambda t, args=None, a=c1, b=c2: a(t, args) + b(t, args))
