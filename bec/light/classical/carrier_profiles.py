from __future__ import annotations

from collections.abc import Callable

import numpy as np
from smef.core.units import Q, QuantityLike, as_quantity, magnitude

# A frequency profile returns delta_omega(t) in rad/s (unitful).
OmegaFn = Callable[[QuantityLike], QuantityLike]


def constant(delta_omega: QuantityLike) -> OmegaFn:
    """
    delta_omega(t) = constant

    Parameters
    ----------
    delta_omega:
        Constant detuning in rad/s (unitful).
    """
    dw = as_quantity(delta_omega, "rad/s")

    def fn(_t: QuantityLike) -> QuantityLike:
        return dw

    return fn


def linear_chirp(
    *, rate: QuantityLike, t0: QuantityLike = Q(0.0, "s")
) -> OmegaFn:
    """
    delta_omega(t) = rate * (t - t0)

    rate has units rad/s^2
    """
    r = as_quantity(rate, "rad/s^2")
    t0q = as_quantity(t0, "s")

    r_val = float(magnitude(r, "rad/s^2"))

    def fn(t: QuantityLike) -> QuantityLike:
        # signed time difference in seconds
        dt_s = float(magnitude(as_quantity(t, "s") - t0q, "s"))
        return Q(r_val * dt_s, "rad/s")

    return fn


def tanh_chirp(
    *, t0: QuantityLike, delta_max: QuantityLike, tau: QuantityLike
) -> OmegaFn:
    """
    delta_omega(t) = delta_max * tanh((t - t0)/tau)
    """
    t0q = as_quantity(t0, "s")
    dm = as_quantity(delta_max, "rad/s")
    tauq = as_quantity(tau, "s")

    dm_val = float(magnitude(dm, "rad/s"))
    tau_s = float(magnitude(tauq, "s"))
    if tau_s <= 0.0:
        raise ValueError("tau must be > 0")

    def fn(t: QuantityLike) -> QuantityLike:
        dt_s = float(magnitude(as_quantity(t, "s") - t0q, "s"))  # signed float
        x = dt_s / tau_s
        return Q(dm_val * float(np.tanh(x)), "rad/s")

    return fn
