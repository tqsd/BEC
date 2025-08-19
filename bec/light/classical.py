from dataclasses import dataclass
from typing import Callable, Optional, Union, Protocol

import numpy as np
from bec.light.envelopes import GaussianEnvelope


class OmegaFn(Protocol):
    """Callable protocol for a QuTiP time coefficient."""

    def __call__(self, t: float, args: Optional[dict] = None) -> float: ...


@dataclass(frozen=True)
class ClassicalTwoPhotonDrive:
    r"""
    Semiclassical two-photon drive (G ↔ XX) with detuning:

        H_drive(t) = Ω(t) (|G⟩⟨XX| + |XX⟩⟨G|) + Δ_L |XX⟩⟨XX|

    with ħ ≡ 1 in the solver.

    Parameters
    ----------
    omega : callable
        Time-dependent Rabi frequency Ω(t) in rad/s. Any callable is accepted.
        Both signatures `f(t)` and `f(t, args)` are supported. A
        `GaussianEnvelope` instance is also valid since it's callable.
    detuning : float
        Two-photon detuning Δ_L in rad/s (constant).
    label : str, optional
        Optional name for bookkeeping/logging.

    Attributes
    ----------
    omega : callable
        The user-provided Ω(t).
    detuning : float
        The user-provided Δ_L (rad/s).
    label : str | None
        Optional label.

    Notes
    -----
    • Use `qutip_coeff()` to obtain a QuTiP-compatible time coefficient
      (it gracefully handles both `f(t)` and `f(t, args)` styles).
    • For a quick setup with a Gaussian envelope, use `from_gaussian(...)`.

    Examples
    --------
    >>> env = GaussianEnvelope(t0=2e-9, sigma=0.5e-9)   # callable: env(t)
    >>> drive = ClassicalTwoPhotonDrive.from_gaussian(env, omega0=2e9, detuning=0.0)
    >>> coeff = drive.qutip_coeff()  # pass as the coefficient for the flip block
    """

    omega: Union[OmegaFn, GaussianEnvelope, Callable[[float], float]]
    detuning: float
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if not callable(self.omega):
            raise TypeError("`omega` must be callable: Ω(t) or Ω(t, args).")
        if not np.isfinite(self.detuning):
            raise ValueError("`detuning` must be a finite float (rad/s).")

    def qutip_coeff(self) -> Callable[[float, Optional[dict]], float]:
        """
        Return a QuTiP-compatible coefficient function: (t, args) -> Ω(t).

        Handles both user callables that accept only `t` and callables that
        accept `(t, args)`.
        """
        f = self.omega

        def coeff(t: float, args: Optional[dict] = None) -> float:
            try:
                return f(t, args)  # type: ignore[misc]
            except TypeError:
                return f(t)  # type: ignore[misc]

        return coeff

    @classmethod
    def from_gaussian(
        cls,
        envelope: GaussianEnvelope,
        omega0: float,
        detuning: float = 0.0,
        label: Optional[str] = None,
    ) -> "ClassicalTwoPhotonDrive":
        """
        Convenience constructor with Ω(t) = Ω0 · envelope(t).

        Parameters
        ----------
        envelope : GaussianEnvelope
            Pulse envelope (callable) giving the unitless shape vs. time.
        omega0 : float
            Peak Rabi frequency Ω0 in rad/s.
        detuning : float, optional
            Δ_L in rad/s (default 0.0).
        label : str, optional
            Optional label.
        """

        def Omega(t: float, args: Optional[dict] = None) -> float:
            return omega0 * envelope(t)

        return cls(omega=Omega, detuning=detuning, label=label)
