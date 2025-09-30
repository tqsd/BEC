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
    """
    Two-photon drive with optional detuning and JSON IO.

    Definition:
    -----------
    A drive is defined by an envelope function f(t) and an amplitude
    `omega0`. The QuTiP cofficient used in time-dependent Hamiltonians
    is produced by `qutip_coeff`, which returns a callable in solver
    time units.

    Serialization:
    --------------
    If the envelope is an instance of `SerializableEnvelope` the
    object can be serialized with `to_dict` and reconstructed with
    `from_dict`. For raf callables, use `from_callablen`.


    Attributes:
    -----------
    envelope: Envelope
        Time-domain function f(t_phys) returning a float.
    omega0: float
        Scalar amplitude multiplier for the envelope value.
    detuning : float or DetuningFn
        Two-photon detuning in rad/s (float)
    label : str or None
        Optional human readable label
    laser_omega: float
        Laser angular frequency
    _cached_tlist: np.ndarray
        Cached solver time grid
    _raw_callable: Callable
        Original callable passed via the `from_callable`
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

    def qutip_coeff(self, *, time_unit_s: float = 1.0) -> OmegaFn:
        """
        Returnst a QuTiP compatible coefficient function

        The solver time is passed by `time_unit_s`.
        """
        env = self.envelope
        om0 = float(self.omega0)
        s = float(time_unit_s)

        def coeff(t: float, args: Optional[dict] = None) -> float:
            t_phys = s * t
            return s * om0 * float(env(t_phys))  # rad per solver-unit

        return coeff

    def detuning_solver(self, *, time_unit_s: float = 1.0) -> float:
        """
        Returns detuning in solver units

        Parameters:
        -----------
        time_unit_s: float
            Solver time unit scaling

        Returns:
        --------
        float
            Solver time unit
        """
        return float(time_unit_s) * float(self.detuning)

    def sample_solver(
        self, tlist_solver: np.ndarray, *, time_unit_s: float
    ) -> np.ndarray:
        """
        Sample Omega_solver(t_prime) on a solver grid

        Arguments:
        ----------
        tlist_solver: np.ndarray
            1D array of solver times
        time_unit_s: seconds per solver time unit.

        Returns:
        --------
        np.ndarray
            1D numpy array with coefficents evaluated on thlist_solver
        """
        coeff = self.qutip_coeff(time_unit_s=time_unit_s)
        return np.array([coeff(t, {}) for t in tlist_solver], dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the drive to a JSON serializable dictionary.

        Returns:
        --------
        dict[str, Any]
            Serializable dictonary

        Raises:
        -------
        TypeError
            If the envelope is not serializable.
        """
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
        """
        Creates a drive from a dictionary produced by `to_dict`.

        Arguments:
        ----------
        data: dict[str, Any]
            Serializable dictionary produced by `to_dict` mehtod.
        Notes:
        ------
        Uses `envelope_from_json` to reconstruct the envelope
        """
        env = envelope_from_json(data["envelope"])
        return cls(
            envelope=env,
            omega0=float(data.get("omega0", 1.0)),
            detuning=float(data.get("detuning", 0.0)),
            label=data.get("label"),
            laser_omega=data.get("laser_omega"),
        )

    @classmethod
    def from_envelope_json(
        cls,
        env_json: Dict[str, Any],
        *,
        omega0: float,
        detuning: float = 0.0,
        laser_omega: float = 0.0,
        label: Optional[str] = None,
    ) -> "ClassicalTwoPhotonDrive":
        """
        Creates a drive from an already serialized envelope JSON.

        Arguments:
        ----------
        env_json: dict[str, Any]
            JSON for a specific envelope
        omega0: float
            amplitude multiplier.
        detuning: float
            two-photon detuning.
        laser_omega: float
            central laser angular frequency
        label: str
            Optional label

        Returns:
        --------
        ClassicalTwoPhotonDrive
        """
        env = envelope_from_json(env_json)
        return cls(
            envelope=env,
            omega0=omega0,
            detuning=detuning,
            label=label,
            laser_omega=laser_omega,
        )

    @classmethod
    def from_callable(
        cls,
        omega_fn: Callable[..., float],
        *,
        detuning: float = 0.0,
        label: Optional[str] = None,
    ) -> "ClassicalTwoPhotonDrive":
        """
        Creates a drive from raw callable.

        Arguments:
        ----------
        omega_fn: Callable
            Callable representing the envelope
        detuning: float
            Two photon detuning
        label: Optional[str]
            Optional string label

        Returns:
        --------
        ClassicalTwoPhotonDrive

        Notes:
        ------
        The callable is wrapped into a simple envelope which
        is not JSON serializable.
        """

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
        """
        Returns a new instance with an updated cached time list.

        Arguments:
        ----------
        tlist: np.ndarray
            1D numpy array for solver times to cache

        Returns:
        --------
        new ClassicalTwoPhotonDrive
            with `_cached_tlist` replaced
        """
        return replace(self, _cached_tlist=np.array(tlist, copy=True))

    def with_detuning(
        self, det: Union[float, DetuningFn]
    ) -> "ClassicalTwoPhotonDrive":
        """
        Returns a new instance with a different detuning.

        Arguments:
        ----------
        det: float
            Detuning parameter

        Returns:
        --------
        new ClassicalTwoPhotonDrive
            with detuning replaced
        """
        return replace(self, detuning=det)
