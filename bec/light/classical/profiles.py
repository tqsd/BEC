from __future__ import annotations

from typing import Callable

import numpy as np

from bec.units import QuantityLike, as_quantity, magnitude, Q


OmegaFn = Callable[[QuantityLike], QuantityLike]


def constant(delta_omega: QuantityLike) -> OmegaFn:
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

    def fn(t: QuantityLike) -> QuantityLike:
        dt_s = magnitude(as_quantity(t, "s") - t0q, "s")
        return Q(float(magnitude(r, "rad/s^2")) * dt_s, "rad/s")

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

    def fn(t: QuantityLike) -> QuantityLike:
        dt = as_quantity(t, "s") - t0q
        dt_s = float(dt.to("s").magnitude)  # signed float
        x = dt_s / tau_s
        return Q(dm_val * float(np.tanh(x)), "rad/s")

    return fn
