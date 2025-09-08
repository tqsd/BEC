from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
    Dict,
    Any,
    Type,
    Union,
)
import numpy as np


# ---------- Envelope Protocols ----------


@runtime_checkable
class Envelope(Protocol):
    """Time-dependent envelope callable: f(t) -> float."""

    def __call__(self, t: float) -> float: ...


@runtime_checkable
class SerializableEnvelope(Envelope, Protocol):
    """Envelope that can be serialized to/from JSON-compatible dict."""

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SerializableEnvelope: ...


# ---------- Concrete Envelopes (Serializable) ----------


@dataclass(frozen=True)
class GaussianEnvelope(SerializableEnvelope):
    """
    Normalized Gaussian envelope:
        f(t) = (area / (σ √(2π))) * exp(-(t - t0)^2 / (2 σ^2))
    Ensures ∫ f(t) dt = area.
    """

    t0: float
    sigma: float
    area: float

    def __call__(self, t: float) -> float:
        norm = self.area / (self.sigma * np.sqrt(2.0 * np.pi))
        return norm * np.exp(-0.5 * ((t - self.t0) / self.sigma) ** 2)

    @classmethod
    def from_fwhm(cls, t0: float, fwhm: float, area: float) -> GaussianEnvelope:
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return cls(t0=t0, sigma=sigma, area=area)

    # --- serialization ---
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "gaussian",
            "t0": self.t0,
            "sigma": self.sigma,
            "area": self.area,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GaussianEnvelope:
        return cls(
            t0=float(data["t0"]),
            sigma=float(data["sigma"]),
            area=float(data["area"]),
        )


@dataclass(frozen=True)
class TPEGaussianEnvelope(SerializableEnvelope):
    """
    Effective two-photon Gaussian envelope:
        f(t) = strength * (area / (σ √(2π))) * exp(-(t - t0)^2 / (2 σ^2))
    """

    t0: float
    sigma: float
    area: float
    strength: float = 1.0

    def __call__(self, t: float) -> float:
        norm = self.area / (self.sigma * np.sqrt(2.0 * np.pi))
        return (
            self.strength
            * norm
            * np.exp(-0.5 * ((t - self.t0) / self.sigma) ** 2)
        )

    @classmethod
    def from_fwhm(
        cls, t0: float, fwhm: float, area: float, strength: float = 1.0
    ) -> TPEGaussianEnvelope:
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return cls(t0=t0, sigma=sigma, area=area, strength=strength)

    # --- serialization ---
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tpe_gaussian",
            "t0": self.t0,
            "sigma": self.sigma,
            "area": self.area,
            "strength": self.strength,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TPEGaussianEnvelope:
        return cls(
            t0=float(data["t0"]),
            sigma=float(data["sigma"]),
            area=float(data["area"]),
            strength=float(data.get("strength", 1.0)),
        )


@dataclass(frozen=True)
class TabulatedEnvelope(SerializableEnvelope):
    """
    Arbitrary shape via samples (piecewise-linear interpolation).

    Args
    ----
    t: strictly increasing sample times [s]
    y: corresponding values (same length as t)

    Notes
    -----
    - This is the safest way to ship arbitrary user-defined envelopes over JSON.
    - Use enough samples for your desired accuracy.
    """

    t: tuple[float, ...]
    y: tuple[float, ...]

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

    # --- serialization ---
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "tabulated", "t": list(self.t), "y": list(self.y)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TabulatedEnvelope:
        t = tuple(float(v) for v in data["t"])
        y = tuple(float(v) for v in data["y"])
        return cls(t=t, y=y)


# ---------- Envelope Registry & JSON helpers ----------

ENVELOPE_REGISTRY: Dict[str, Type[SerializableEnvelope]] = {
    "gaussian": GaussianEnvelope,
    "tpe_gaussian": TPEGaussianEnvelope,
    "tabulated": TabulatedEnvelope,
}


def envelope_to_json(env: SerializableEnvelope) -> Dict[str, Any]:
    """Serialize a SerializableEnvelope to a JSON-ready dict."""
    return env.to_dict()


def envelope_from_json(data: Dict[str, Any]) -> SerializableEnvelope:
    """Construct a SerializableEnvelope from its JSON-ready dict."""
    t = data.get("type")
    if not isinstance(t, str):
        raise ValueError("Envelope JSON must contain a string 'type' field.")
    cls = ENVELOPE_REGISTRY.get(t)
    if cls is None:
        raise ValueError(
            f"Unknown envelope type '{t}'. Known: {list(ENVELOPE_REGISTRY)}"
        )
    return cls.from_dict(data)


# ---------- Drive (generalized) ----------


class OmegaFn(Protocol):
    """QuTiP-compatible coefficient: (t, args) -> float."""

    def __call__(self, t: float, args: Optional[dict] = None) -> float: ...


@dataclass(frozen=True)
class ClassicalTwoPhotonDrive:
    r"""
    Two-photon drive (G ↔ XX) with detuning:

        H_drive(t) = Ω(t) (|G⟩⟨XX| + |XX⟩⟨G|) + Δ_L |XX⟩⟨XX|

    Accepts:
      • a plain omega callable Ω(t[, args]),
      • an Envelope (callable f(t)), optionally scaled by `omega0`,
      • or a JSON dict describing a SerializableEnvelope.

    If `envelope` is provided, Ω(t) = omega0 * envelope(t).
    If a plain `omega` is provided, `envelope`/`omega0` are ignored.
    """

    # Mutually exclusive in practice: user gives `omega` OR (`envelope` + optional `omega0`)
    omega: Optional[Callable[..., float]] = None
    envelope: Optional[Envelope] = None
    omega0: float = 1.0
    detuning: float = 0.0
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.omega is None and self.envelope is None:
            raise ValueError("Provide either `omega` or `envelope`.")
        if self.omega is not None and not callable(self.omega):
            raise TypeError("`omega` must be callable.")
        if not np.isfinite(self.detuning):
            raise ValueError("`detuning` must be finite.")

    def qutip_coeff(self) -> OmegaFn:
        """
        Return QuTiP-compatible coeff: (t, args) -> Ω(t).
        """
        if self.omega is not None:
            f = self.omega

            def coeff(t: float, args: Optional[dict] = None) -> float:
                try:
                    return float(f(t, args))  # type: ignore[misc]
                except TypeError:
                    return float(f(t))  # type: ignore[misc]

            return coeff

        env = self.envelope
        omega0 = float(self.omega0)

        def coeff(t: float, args: Optional[dict] = None) -> float:
            return omega0 * float(env(t))  # type: ignore[arg-type]

        return coeff

    # ----- convenience constructors -----

    @classmethod
    def from_envelope(
        cls,
        envelope: Envelope,
        omega0: float,
        detuning: float = 0.0,
        label: Optional[str] = None,
    ) -> ClassicalTwoPhotonDrive:
        return cls(
            omega=None,
            envelope=envelope,
            omega0=omega0,
            detuning=detuning,
            label=label,
        )

    @classmethod
    def from_envelope_json(
        cls,
        env_json: Dict[str, Any],
        omega0: float,
        detuning: float = 0.0,
        label: Optional[str] = None,
    ) -> ClassicalTwoPhotonDrive:
        env = envelope_from_json(env_json)
        return cls.from_envelope(
            env, omega0=omega0, detuning=detuning, label=label
        )

    @classmethod
    def from_omega_callable(
        cls,
        omega: Callable[..., float],
        detuning: float = 0.0,
        label: Optional[str] = None,
    ) -> ClassicalTwoPhotonDrive:
        return cls(
            omega=omega,
            envelope=None,
            omega0=1.0,
            detuning=detuning,
            label=label,
        )


def gaussian_to_tabulated(
    *,
    t0: float,
    sigma: Optional[float] = None,
    fwhm: Optional[float] = None,
    area: float = np.pi,
    nsigma: float = 6.0,  # window half-width = nsigma * sigma
    num: int = 401,  # samples (odd keeps a sample at t0)
    to_json: bool = True,  # return JSON spec ready for your parser
) -> Union[Dict[str, Any], TabulatedEnvelope]:
    """Tabulate a normalized Gaussian and return a JSON envelope spec."""
    if sigma is None:
        if fwhm is None:
            raise ValueError("Provide either sigma or fwhm.")
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    t_lo = t0 - nsigma * sigma
    t_hi = t0 + nsigma * sigma
    t = np.linspace(t_lo, t_hi, num)

    # Analytic normalized Gaussian with ∫Ω dt = area (on infinite support)
    norm = area / (sigma * np.sqrt(2.0 * np.pi))
    y = norm * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    # Renormalize because we truncated to [t_lo, t_hi] and discretized
    trap = np.trapezoid(y, t)
    if trap > 0:
        y *= area / trap

    if to_json:
        # JSON-ready dict for your `envelope_from_json`
        return {
            "type": "tabulated",
            "t": [float(v) for v in t],
            "y": [float(v) for v in y],
        }
    else:
        # If you want the object directly:
        return TabulatedEnvelope(
            t=tuple(float(v) for v in t), y=tuple(float(v) for v in y)
        )
