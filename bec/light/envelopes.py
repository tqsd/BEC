# bec/light/envelopes.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Dict, Any, Type, Tuple
import numpy as np
import math

# ---------- Protocols ----------


@runtime_checkable
class Envelope(Protocol):
    """Time-dependent envelope: f(t) -> float."""

    def __call__(self, t: float) -> float: ...


@runtime_checkable
class SerializableEnvelope(Envelope, Protocol):
    """Envelope that can be serialized to/from a JSON-compatible dict."""

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SerializableEnvelope: ...


# ---------- Concrete envelopes ----------


@dataclass(frozen=True)
class GaussianEnvelope(SerializableEnvelope):
    r"""Normalized Gaussian:
    :math:`f(t) = \frac{\text{area}}{\sigma \sqrt{2\pi}} \exp\!\left(-\frac{(t-t_0)^2}{2\sigma^2}\right)`.
    Ensures :math:`\int f(t)\,dt = \text{area}` on :math:`(-\infty,\infty)`.
    """

    t0: float
    sigma: float
    area: float

    def __call__(self, t: float) -> float:
        norm = self.area / (self.sigma * math.sqrt(2.0 * math.pi))
        x = (t - self.t0) / self.sigma
        return float(norm * math.exp(-0.5 * x * x))

    @classmethod
    def from_fwhm(
        cls, t0: float, fwhm: float, area: float
    ) -> "GaussianEnvelope":
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return cls(t0=t0, sigma=sigma, area=area)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "gaussian",
            "t0": self.t0,
            "sigma": self.sigma,
            "area": self.area,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaussianEnvelope":
        return cls(
            t0=float(data["t0"]),
            sigma=float(data["sigma"]),
            area=float(data["area"]),
        )


@dataclass(frozen=True)
class TabulatedEnvelope(SerializableEnvelope):
    """Arbitrary shape via samples (piecewise-linear interpolation)."""

    t: Tuple[float, ...]
    y: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.t) != len(self.y) or len(self.t) < 2:
            raise ValueError("t and y must have the same length >= 2.")
        if any(np.isnan(self.t)) or any(np.isnan(self.y)):
            raise ValueError("NaNs in t or y are not allowed.")
        if any(self.t[i] >= self.t[i + 1] for i in range(len(self.t) - 1)):
            raise ValueError("t must be strictly increasing.")

    def __call__(self, t: float) -> float:
        return float(
            np.interp(t, self.t, self.y, left=self.y[0], right=self.y[-1])
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "tabulated", "t": list(self.t), "y": list(self.y)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabulatedEnvelope":
        t = tuple(float(v) for v in data["t"])
        y = tuple(float(v) for v in data["y"])
        return cls(t=t, y=y)


@dataclass(frozen=True)
class SymbolicEnvelope(SerializableEnvelope):
    r"""Symbolic expression in terms of ``t`` and parameters.

    Example:
        expr = "A * exp(-((t - t0)**2) / (2*sigma**2)) * cos(omega*t + phi)"
        params = {"A": 1.0, "t0": 0.0, "sigma": 20e-12, "omega": 2*np.pi*10e9, "phi": 0.0}

    Notes:
        For safety and speed we evaluate with a restricted namespace containing
        Python's ``math`` and ``numpy`` under ``np``. No ``eval`` on globals.
    """

    expr: str
    params: Dict[str, float]

    def __call__(self, t: float) -> float:
        local = {"t": float(t), "np": np, "math": math}
        local.update(self.params)
        # controlled env
        return float(eval(self.expr, {"__builtins__": {}}, local))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "symbolic",
            "expr": self.expr,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolicEnvelope":
        expr = str(data["expr"])
        params = {
            str(k): float(v) for k, v in dict(data.get("params", {})).items()
        }
        return cls(expr=expr, params=params)


# ---------- Registry ----------


ENVELOPE_REGISTRY: Dict[str, Type[SerializableEnvelope]] = {
    "gaussian": GaussianEnvelope,
    "tabulated": TabulatedEnvelope,
    "symbolic": SymbolicEnvelope,
}


def envelope_to_json(env: SerializableEnvelope) -> Dict[str, Any]:
    """Serialize to a JSON-ready dict."""
    return env.to_dict()


def envelope_from_json(data: Dict[str, Any]) -> SerializableEnvelope:
    """Construct from a JSON-ready dict."""
    t = data.get("type")
    if not isinstance(t, str):
        raise ValueError("Envelope JSON must contain a string 'type' field.")
    cls = ENVELOPE_REGISTRY.get(t)
    if cls is None:
        raise ValueError(
            f"Unknown envelope type '{
                         t}'. Known: {list(ENVELOPE_REGISTRY)}"
        )
    return cls.from_dict(data)
