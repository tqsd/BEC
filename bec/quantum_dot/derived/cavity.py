from __future__ import annotations

from functools import cached_property
import numpy as np

from bec.units import QuantityLike, Q, as_quantity, c


class CavityMixin:
    @cached_property
    def cavity_lambda(self) -> QuantityLike:
        cp = getattr(self.qd, "cavity_params", None)
        if cp is None:
            return Q(0.0, "m")
        if hasattr(cp, "lambda_m"):
            return as_quantity(cp.lambda_m, "m")
        return as_quantity(cp.lambda_cav, "m")

    @cached_property
    def cavity_omega(self) -> QuantityLike:
        lam = self.cavity_lambda
        if float(lam.to("m").magnitude) == 0.0:
            return Q(0.0, "rad/s")
        nu = (c / lam).to("Hz")
        return (2.0 * np.pi * nu).to("rad/s")

    @cached_property
    def cavity_kappa(self) -> QuantityLike:
        cp = getattr(self.qd, "cavity_params", None)
        if cp is None:
            return Q(0.0, "rad/s")
        Qfac = float(cp.Q)
        if Qfac <= 0:
            return Q(0.0, "rad/s")
        return (self.cavity_omega / Qfac).to("rad/s")
