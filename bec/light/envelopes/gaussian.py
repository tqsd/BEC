from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import math

from bec.units import QuantityLike, as_quantity, magnitude

from .base import SerializableEnvelope, TimeLike, _time_s


@dataclass(frozen=True)
class GaussianEnvelope(SerializableEnvelope):
    """
    Peak-normalized Gaussian envelope.

    g(t) = exp(-(t - t0)^2 / (2*sigma^2))

    max g(t) = 1
    """

    t0: QuantityLike
    sigma: QuantityLike

    _t0_s: float = field(init=False, repr=False)
    _sig_s: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        t0q = as_quantity(self.t0, "s")
        sigq = as_quantity(self.sigma, "s")

        t0_s = float(t0q.magnitude)
        sig_s = float(sigq.magnitude)
        if sig_s <= 0.0:
            raise ValueError("sigma must be > 0")

        object.__setattr__(self, "t0", t0q)
        object.__setattr__(self, "sigma", sigq)
        object.__setattr__(self, "_t0_s", t0_s)
        object.__setattr__(self, "_sig_s", sig_s)

    def _eval_seconds(self, t_s: float) -> float:
        x = (t_s - self._t0_s) / self._sig_s
        return float(math.exp(-0.5 * x * x))

    def __call__(self, t: TimeLike) -> float:
        return self._eval_seconds(_time_s(t))

    def area_seconds(self) -> float:
        """Return integral of g(t) dt in seconds."""
        return self._sig_s * math.sqrt(2.0 * math.pi)

    @classmethod
    def from_fwhm(cls, t0: Any, fwhm: Any) -> "GaussianEnvelope":
        t0_q = as_quantity(t0, "s")
        fwhm_q = as_quantity(fwhm, "s")
        fwhm_s = magnitude(fwhm_q, "s")
        if fwhm_s <= 0.0:
            raise ValueError("fwhm must be > 0")
        sigma_s = fwhm_s / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return cls(t0=t0_q, sigma=as_quantity(sigma_s, "s"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "gaussian",
            "t0": {"value": float(magnitude(self.t0, "s")), "unit": "s"},
            "sigma": {"value": float(magnitude(self.sigma, "s")), "unit": "s"},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaussianEnvelope":
        t0_d = data["t0"]
        sig_d = data["sigma"]
        t0 = as_quantity(float(t0_d["value"]), str(t0_d.get("unit", "s")))
        sigma = as_quantity(float(sig_d["value"]), str(sig_d.get("unit", "s")))
        return cls(t0=t0, sigma=sigma)
