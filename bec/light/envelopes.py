from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class EnvelopeFn(Protocol):
    """
    Protocol for time-dependent envelopes used as couplings in Hamiltonians.

    Any envelope must be callable as f(t: float) -> float and be immutable
    (dataclass(frozen=True) recommended).
    """

    def __call__(self, t: float) -> float: ...


@dataclass(frozen=True)
class GaussianEnvelope:
    """
    Normalized Gaussian pulse Ω(t) for single-photon coupling.

    Definition
    ----------
    Ω(t) = (area / (σ √(2π))) * exp(-(t - t0)^2 / (2 σ^2))

    Parameters
    ----------
    t0 : float
        Pulse center time [s].
    sigma : float
        Standard deviation (temporal width) [s].
    area : float
        Integrated pulse area ∫ Ω(t) dt [rad]. E.g., π for a π-pulse.

    Notes
    -----
    - The normalization guarantees ∫_{-∞}^{∞} Ω(t) dt = area.
    - Use this directly as a QuTiP time-dependent coefficient: it is callable.
    """

    t0: float
    sigma: float
    area: float

    def __call__(self, t: float) -> float:
        norm = self.area / (self.sigma * np.sqrt(2.0 * np.pi))
        return norm * np.exp(-0.5 * ((t - self.t0) / self.sigma) ** 2)

    @classmethod
    def from_fwhm(
        cls, t0: float, fwhm: float, area: float
    ) -> "GaussianEnvelope":
        """
        Construct from FWHM [s] instead of sigma.  sigma = fwhm / (2√(2 ln 2)).
        """
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return cls(t0=t0, sigma=sigma, area=area)


@dataclass(frozen=True)
class TPEGaussianEnvelope:
    """
    Effective two-photon Gaussian coupling Λ(t) for TPE Hamiltonian.

    Definition
    ----------
    Λ(t) = (area / (σ √(2π))) * exp(-(t - t0)^2 / (2 σ^2)) * strength

    Parameters
    ----------
    t0 : float
        Pulse center time [s].
    sigma : float
        Standard deviation (temporal width) [s].
    area : float
        Integrated *effective* two-photon area ∫ Λ(t) dt [rad].
        (Choose based on desired |G⟷XX| population dynamics.)
    strength : float
        Dimensionless scaling to fold in constant factors
        (e.g., ∑ g1 g2 / Δ_i).
        Keep at 1.0 if you handle that elsewhere.

    Notes
    -----
    - Same callable interface as GaussianEnvelope so you can pass either
      to LightMode. This represents the *effective* Λ(t) after adiabatic
      elimination; it does not compute detuning-dependent prefactors itself.
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
    ) -> "TPEGaussianEnvelope":
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return cls(t0=t0, sigma=sigma, area=area, strength=strength)


# Optional convenience alias for annotations
EnvelopeLike = EnvelopeFn | GaussianEnvelope | TPEGaussianEnvelope
