# bec/light/classical.py

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, Callable, Dict, Any, Protocol, Union
import numpy as np
from bec.light.envelopes import (
    Envelope,
    SerializableEnvelope,
    envelope_from_json,
    envelope_to_json,
)


class OmegaFn(Protocol):
    """QuTiP-compatible coefficient: (t, args) -> float."""

    def __call__(self, t: float, args: Optional[dict] = None) -> float: ...


DetuningFn = Callable[[float], float]


@dataclass(frozen=True)
class ClassicalTwoPhotonDrive:
    r"""Two-photon drive with detuning.

    Mathematics:
        Effective coefficient:
        :math:`\Omega(t) = \Omega_0 \, f(t)`,
        where :math:`f(t)` is the envelope and :math:`\Omega_0` is a scalar.

    Args:
        envelope: Time-domain envelope (callable); if serializable, it can be saved/loaded.
        omega0: Scalar multiplier for the envelope value.
        detuning: Two-photon detuning (rad/s).
        label: Optional name.

    Notes:
        - For QuTiP, use :meth:`qutip_coeff` to get a ``(t, args)->float`` coefficient.
        - Use :meth:`sample` to get a vectorized sample on a given ``tlist``.
        - Use :meth:`to_dict` / :meth:`from_dict` for JSON IO (if the envelope is serializable).
    """

    envelope: Envelope
    omega0: float = 1.0
    detuning: Union[float, DetuningFn] = 0.0
    label: Optional[str] = None
    laser_omega: Optional[float] = None
    _cached_tlist: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False, compare=False
    )

    # optional legacy path for raw callables (kept here but not encouraged)
    _raw_callable: Optional[Callable[..., float]] = None

    # --------- main API ---------

    def qutip_coeff(self, *, time_unit_s: float = 1.0) -> OmegaFn:
        """Return (t', args)->Ω_solver(t') with Ω_solver = time_unit_s * Ω_phys(t_phys),
        where t_phys = time_unit_s * t'.
        """
        env = self.envelope
        om0 = float(self.omega0)
        s = float(time_unit_s)

        def coeff(t: float, args: Optional[dict] = None) -> float:
            t_phys = s * t
            return s * om0 * float(env(t_phys))  # rad per solver-unit

        return coeff

    def detuning_solver(self, *, time_unit_s: float = 1.0) -> float:
        """Δ_solver = time_unit_s * Δ_phys (rad per solver-unit)."""
        return float(time_unit_s) * float(self.detuning)

    # (keep your sample(...) if you like, but be explicit about which units you want)
    def sample_solver(
        self, tlist_solver: np.ndarray, *, time_unit_s: float
    ) -> np.ndarray:
        """Ω_solver(t') sampled on the solver grid."""
        coeff = self.qutip_coeff(time_unit_s=time_unit_s)
        return np.array([coeff(t, {}) for t in tlist_solver], dtype=float)

    # --------- JSON IO (only if the envelope is serializable) ---------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the drive; raises if the envelope is not serializable."""
        if not isinstance(self.envelope, SerializableEnvelope):
            raise TypeError("Envelope is not serializable; cannot to_dict().")
        return {
            "type": "classical_2photon",
            "label": self.label,
            "omega0": float(self.omega0),
            "detuning": float(self.detuning),
            "envelope": envelope_to_json(self.envelope),
            "laser_omega": (
                None if self.laser_omega is None else float(self.laser_omega)
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassicalTwoPhotonDrive":
        env = envelope_from_json(data["envelope"])
        return cls(
            envelope=env,
            omega0=float(data.get("omega0", 1.0)),
            detuning=float(data.get("detuning", 0.0)),
            label=data.get("label"),
            laser_omega=data.get("laser_omega"),
        )

    # --------- convenience constructors ---------

    @classmethod
    def from_envelope_json(
        cls,
        env_json: Dict[str, Any],
        *,
        omega0: float,
        detuning: float = 0.0,
        label: Optional[str] = None,
    ) -> "ClassicalTwoPhotonDrive":
        env = envelope_from_json(env_json)
        return cls(envelope=env, omega0=omega0, detuning=detuning, label=label)

    @classmethod
    def from_callable(
        cls,
        omega_fn: Callable[..., float],
        *,
        detuning: float = 0.0,
        label: Optional[str] = None,
    ) -> "ClassicalTwoPhotonDrive":
        # Wrap a raw callable as an envelope
        def env(t: float) -> float:
            try:
                return float(omega_fn(t, None))  # type: ignore[misc]
            except TypeError:
                return float(omega_fn(t))  # type: ignore[misc]

        return cls(
            envelope=env,
            omega0=1.0,
            detuning=detuning,
            label=label,
            _raw_callable=omega_fn,
        )

    def with_cached_tlist(self, tlist: np.ndarray) -> "ClassicalTwoPhotonDrive":
        return replace(self, _cached_tlist=np.array(tlist, copy=True))

    def with_detuning(
        self, det: Union[float, DetuningFn]
    ) -> "ClassicalTwoPhotonDrive":
        return replace(self, detuning=det)
