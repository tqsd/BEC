from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Dict, Any, Type, Tuple
import numpy as np
import math


@runtime_checkable
class Envelope(Protocol):
    """
    Callable time-dependent envelope.

    Contract:
      f(t: float) -> float

    The value should be finite for all relevant t, expressed in seconds.
    """

    def __call__(self, t: float) -> float: ...


@runtime_checkable
class SerializableEnvelope(Envelope, Protocol):
    """
    Envelope that supports JSON serialization.

    Required methods:
    -----------------
     - `to_dict() -> dict[str, Any]`
     - `from_dict(data: dict[str, Any]) -> SerializableEnvelope`

    Implementations must include a "type" field that can be used
    to look up the constructor in ENVELOPE_REGISTRY
    """

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SerializableEnvelope: ...


@dataclass(frozen=True)
class GaussianEnvelope(SerializableEnvelope):
    r"""
    Normalized Gaussian envelope

    Definition:
    -----------

    .. math::
      f(t) = A / (\sigma * \sqrt{2*\pi}) * \exp(- \frac{(t - t0)^2}{(2*\sigma^2)})

    This normalization ensures the integral over (-inf, inf) equals "area".

    Attributes:
      t0: Center time in seconds.
      sigma: RMS width in seconds (must be > 0).
      area: Time integral of the envelope (same units as f(t) * s).

    Notes:
      Use from_fwhm to build from the full-width at half maximum (FWHM).
    """

    t0: float
    sigma: float
    area: float

    def __call__(self, t: float) -> float:
        """
        Evaluate the Gaussian at time t (seconds)
        """
        norm = self.area / (self.sigma * math.sqrt(2.0 * math.pi))
        x = (t - self.t0) / self.sigma
        return float(norm * math.exp(-0.5 * x * x))

    @classmethod
    def from_fwhm(
        cls, t0: float, fwhm: float, area: float
    ) -> "GaussianEnvelope":
        r"""
        Construct from full-width at half maximum (FWHM)

        .. math::
            \sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}

        Arguments:
        ----------
        t0: float
            Center time (s)
        fwhm: float
            full widhth at half maximum, must be > 0
        area: Integral area of the envelope

        Returns:
        --------
        GaussianEnvelope
        """
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return cls(t0=t0, sigma=sigma, area=area)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to JSON ready dictionary
        """
        return {
            "type": "gaussian",
            "t0": self.t0,
            "sigma": self.sigma,
            "area": self.area,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaussianEnvelope":
        """
        Deserialize from a JSON-like dictionary
        """
        return cls(
            t0=float(data["t0"]),
            sigma=float(data["sigma"]),
            area=float(data["area"]),
        )


@dataclass(frozen=True)
class TabulatedEnvelope(SerializableEnvelope):
    """
    Piecewise-linear envelope defined by samples

    Values between samples are linearly interpolated. Values outside
    the range are clamped to the first/last sample value.

    Attributes:
    -----------
    t: Tuple[float]
       Strictly increasing time samples (seconds), len>2
    x: Tuple[float]
       Corresponding values, same length as t
    """

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
        """Evaluate by linear interpolation with clamping at the ends."""
        return float(
            np.interp(t, self.t, self.y, left=self.y[0], right=self.y[-1])
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize
        """
        return {"type": "tabulated", "t": list(self.t), "y": list(self.y)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabulatedEnvelope":
        """
        Deserialize
        """
        t = tuple(float(v) for v in data["t"])
        y = tuple(float(v) for v in data["y"])
        return cls(t=t, y=y)


@dataclass(frozen=True)
class SymbolicEnvelope(SerializableEnvelope):
    """
    Symbolic envelope evaluated via a restricted eval.


    This envelope evaluates an expression string in terms of `t` and
    parameters.

    Example:
    --------
    >>> expr = "A * math.exp(-((t - t0)**2)/(2*sigma**2)) * np.cos(omega*t + phi)"
    >>> params = {"A": 1.0, "t0": 0.0, "sigma": 20e-12, "omega": 2*np.pi*10e9, "phi": 0.0}

    Safety:
    -------
      - Uses a restricted namespace with only {"__builtins__": {}} in globals.
      - Provides local symbols: "t" (float), "math" (Python math), "np" (numpy),
        and the given "params".
      - No access to other globals.

    Attributes:
    -----------
    expr: str
        Python expression string using "t", "math", "np", and params.
    params: dict[str, float]
        Mapping of parameter names to float values.
    """

    expr: str
    params: Dict[str, float]

    def __call__(self, t: float) -> float:
        """evaluate the expression in time (seconds)"""
        local = {"t": float(t), "np": np, "math": math}
        local.update(self.params)
        # controlled env
        return float(eval(self.expr, {"__builtins__": {}}, local))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize"""
        return {
            "type": "symbolic",
            "expr": self.expr,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolicEnvelope":
        """Deserialize"""
        expr = str(data["expr"])
        params = {
            str(k): float(v) for k, v in dict(data.get("params", {})).items()
        }
        return cls(expr=expr, params=params)


ENVELOPE_REGISTRY: Dict[str, Type[SerializableEnvelope]] = {
    "gaussian": GaussianEnvelope,
    "tabulated": TabulatedEnvelope,
    "symbolic": SymbolicEnvelope,
}


def envelope_to_json(env: SerializableEnvelope) -> Dict[str, Any]:
    """Serialize to a JSON-ready dict."""
    return env.to_dict()


def envelope_from_json(data: Dict[str, Any]) -> SerializableEnvelope:
    t = data.get("type")
    if not isinstance(t, str):
        raise ValueError("Envelope JSON must contain a string 'type' field.")
    cls = ENVELOPE_REGISTRY.get(t)
    if cls is None:
        known = ", ".join(sorted(ENVELOPE_REGISTRY.keys()))
        raise ValueError(f"Unknown envelope type '{t}'. Known: {known}")
    return cls.from_dict(data)
