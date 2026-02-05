from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smef.core.units import Q, QuantityLike, as_quantity, magnitude

from .base import SerializableEnvelopeU


def _as_time_quantity(x: Any, unit: str = "s") -> QuantityLike:
    """
    Coerce x into a time quantity in the SMEF unit registry.
    - bare numbers: interpreted as `unit`
    - quantities: converted to `unit` (raises if incompatible)
    """
    return as_quantity(x, unit)


def _mag(x: Any, unit: str) -> float:
    return float(magnitude(x, unit))


@dataclass(frozen=True)
class GaussianEnvelopeU(SerializableEnvelopeU):
    """
    Peak-normalized Gaussian envelope (unitful time).

    g(t) = exp(-(t - t0)^2 / (2*sigma^2))
    max g(t) = 1
    """

    t0: QuantityLike
    sigma: QuantityLike

    _t0_s: float = field(init=False, repr=False)
    _sig_s: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        t0_q = _as_time_quantity(self.t0, "s")
        sig_q = _as_time_quantity(self.sigma, "s")

        t0_s = float(t0_q.magnitude)
        sig_s = float(sig_q.magnitude)
        if sig_s <= 0.0:
            raise ValueError("sigma must be > 0")

        object.__setattr__(self, "t0", t0_q)
        object.__setattr__(self, "sigma", sig_q)
        object.__setattr__(self, "_t0_s", t0_s)
        object.__setattr__(self, "_sig_s", sig_s)

    def _eval_seconds(self, t_s: float) -> float:
        x = (t_s - self._t0_s) / self._sig_s
        return float(math.exp(-0.5 * x * x))

    def __call__(self, t: QuantityLike) -> float:
        # Strictly unitful per your EnvelopeU contract
        if not hasattr(t, "to"):
            raise TypeError("EnvelopeU requires unitful time (QuantityLike).")
        return self._eval_seconds(_mag(t, "s"))

    def area_seconds(self) -> float:
        """Integral of g(t) dt in seconds."""
        return self._sig_s * math.sqrt(2.0 * math.pi)

    @classmethod
    def from_fwhm(cls, t0: Any, fwhm: Any) -> GaussianEnvelopeU:
        t0_q = _as_time_quantity(t0, "s")
        fwhm_q = _as_time_quantity(fwhm, "s")
        fwhm_s = float(fwhm_q.magnitude)
        if fwhm_s <= 0.0:
            raise ValueError("fwhm must be > 0")

        sigma_s = fwhm_s / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return cls(t0=t0_q, sigma=Q(sigma_s, "s"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "gaussian",
            "t0": {"value": _mag(self.t0, "s"), "unit": "s"},
            "sigma": {"value": _mag(self.sigma, "s"), "unit": "s"},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GaussianEnvelopeU:
        t0_d = data["t0"]
        sig_d = data["sigma"]

        t0_v = float(t0_d["value"])
        t0_u = str(t0_d.get("unit", "s"))

        sig_v = float(sig_d["value"])
        sig_u = str(sig_d.get("unit", "s"))

        t0_q = Q(t0_v, t0_u).to("s")
        sig_q = Q(sig_v, sig_u).to("s")
        return cls(t0=t0_q, sigma=sig_q)
