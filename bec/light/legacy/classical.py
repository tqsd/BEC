from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Literal,
    Optional,
    Callable,
    Dict,
    Any,
    Tuple,
    Union,
)
import numpy as np
from scipy.constants import e as _e, hbar as _hbar

from bec.light.envelopes import (
    Envelope,
    GaussianEnvelope,
    SerializableEnvelope,
    envelope_from_json,
    envelope_to_json,
)
from bec.params.energy_levels import EnergyLevels
from bec.params.transitions import Transition


PolBasis = Literal["HV"]

# physical-time function: t_phys [s] -> rad/s
ChirpFn = Callable[[float], float]
AmplitudeKind = Literal["rabi", "field"]


def _c_to_json(z: complex) -> list[float]:
    """Encode a complex number as [re, im] for JSON."""
    return [float(np.real(z)), float(np.imag(z))]


def _c_from_json(x: Any) -> complex:
    """
    Decode a complex number from JSON.
    Accepts:
      - [re, im]
      - (re, im)
      - {"re": ..., "im": ...}  (optional convenience)
      - real numbers (treated as re + 0j)
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return complex(float(x), 0.0)

    if isinstance(x, (list, tuple)) and len(x) == 2:
        return complex(float(x[0]), float(x[1]))

    if isinstance(x, dict) and "re" in x and "im" in x:
        return complex(float(x["re"]), float(x["im"]))

    raise TypeError(f"Cannot decode complex from {type(x)}: {x!r}")


@dataclass(frozen=True)
class JonesState:
    jones: Tuple[complex, complex] = (1 + 0j, 0 + 0j)
    basis: PolBasis = "HV"
    normalize: bool = True

    def as_array(self) -> np.ndarray:
        v = np.array(self.jones, dtype=complex)
        if self.normalize:
            n = np.linalg.norm(v)
            if n != 0:
                v = v / n
        return v

    @classmethod
    def H(cls) -> "JonesState":
        return cls(jones=(1 + 0j, 0 + 0j))

    @classmethod
    def V(cls) -> "JonesState":
        return cls(jones=(0 + 0j, 1 + 0j))

    @classmethod
    def D(cls) -> "JonesState":
        return cls(jones=(1 + 0j, 1 + 0j))

    @classmethod
    def A(cls) -> "JonesState":
        return cls(jones=(1 + 0j, -1 + 0j))

    @classmethod
    def R(cls) -> "JonesState":
        return cls(jones=(1 / np.sqrt(2) + 0j, -1j / np.sqrt(2)))

    @classmethod
    def L(cls) -> "JonesState":
        return cls(jones=(1 / np.sqrt(2) + 0j, 1j / np.sqrt(2)))


@dataclass(frozen=True)
class JonesMatrix:
    J: np.ndarray
    basis: PolBasis = "HV"

    def __post_init__(self) -> None:
        A = np.asarray(self.J, dtype=complex)
        if A.shape != (2, 2):
            raise ValueError("JonesMatrix must be 2x2 complex.")
        object.__setattr__(self, "J", A)

    @classmethod
    def identity(cls, *, basis: PolBasis = "HV") -> "JonesMatrix":
        return cls(J=np.eye(2, dtype=complex), basis=basis)

    @classmethod
    def rotation(cls, theta: float, *, basis: PolBasis = "HV") -> "JonesMatrix":
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        J = np.array([[c, -s], [s, c]], dtype=complex)
        return cls(J=J, basis=basis)

    @classmethod
    def retarder(
        cls,
        delta: float,
        theta: float = 0.0,
        *,
        basis: PolBasis = "HV",
    ) -> "JonesMatrix":
        Rm = cls.rotation(-theta, basis=basis).J
        Rp = cls.rotation(theta, basis=basis).J
        D = np.array(
            [
                [np.exp(-1j * delta / 2.0), 0.0],
                [0.0, np.exp(+1j * delta / 2.0)],
            ],
            dtype=complex,
        )
        return cls(J=Rm @ D @ Rp, basis=basis)

    @classmethod
    def hwp(
        cls, theta: float = 0.0, *, basis: PolBasis = "HV"
    ) -> "JonesMatrix":
        return cls.retarder(np.pi, theta, basis=basis)

    @classmethod
    def qwp(
        cls, theta: float = 0.0, *, basis: PolBasis = "HV"
    ) -> "JonesMatrix":
        return cls.retarder(np.pi / 2.0, theta, basis=basis)

    @classmethod
    def linear_polarizer(
        cls, theta: float = 0.0, *, basis: PolBasis = "HV"
    ) -> "JonesMatrix":
        Rm = cls.rotation(-theta, basis=basis).J
        Rp = cls.rotation(theta, basis=basis).J
        P = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        return cls(J=Rm @ P @ Rp, basis=basis)


@dataclass(frozen=True)
class ClassicalCoherentDrive:
    envelope: Envelope
    # meaning depends on amplitude_kind
    # - rabi: omega0 == peak Rabi scale [rad/s]
    # - field: omega0 == peak E-field scale [V/m]
    omega0: float = 1.0
    amplitude_kind: AmplitudeKind = "rabi"
    pol_state: Optional["JonesState"] = None
    pol_transform: Optional["JonesMatrix"] = None

    laser_omega0: Optional[float] = None
    delta_omega: Union[float, ChirpFn] = 0.0
    label: Optional[str] = None

    _cached_tlist: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False, compare=False
    )
    _raw_callable: Optional[Callable[..., float]] = field(
        default=None, repr=False, compare=False
    )

    def field_phys(self, t_phys: float) -> complex:
        if self.amplitude_kind != "field":
            raise ValueError(
                "field_phys() called but amplitude_kind != 'field'"
            )
        return complex(float(self.omega0) * float(self.envelope(float(t_phys))))

    def omega_phys(self, t_phys: float) -> complex:
        if self.amplitude_kind != "rabi":
            raise ValueError("omega_phys() called but amplitude_kind != 'rabi'")
        return complex(float(self.omega0) * float(self.envelope(float(t_phys))))

    def omega_solver(self, t_solver: float, *, time_unit_s: float) -> complex:
        if self.amplitude_kind != "rabi":
            raise ValueError(
                "omega_solver() is only valid for amplitude_kind='rabi'"
            )
        s = float(time_unit_s)
        t_phys = s * float(t_solver)
        return self.omega_phys(t_phys) * s

    def area_solver(self, tlist: np.ndarray, *, time_unit_s: float) -> float:
        tlist = np.asarray(tlist, dtype=float)
        vals = np.asarray(
            [self.omega_solver(t, time_unit_s=time_unit_s) for t in tlist],
            dtype=np.complex128,
        )
        return float(np.trapezoid(vals.real, tlist))

    def effective_polarization(self) -> Optional[np.ndarray]:
        if self.pol_state is None:
            return None

        if (
            self.pol_transform is not None
            and self.pol_transform.basis != self.pol_state.basis
        ):
            raise ValueError(
                "Polarization basis mismatch: pol_state.basis != pol_transform.basis"
            )

        E = self.pol_state.as_array()
        if self.pol_transform is not None:
            E = self.pol_transform.J @ E
        return E

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassicalCoherentDrive":
        env = envelope_from_json(data["envelope"])

        pol_state = None
        ps = data.get("pol_state")
        if ps is not None:
            pol_state = JonesState(
                jones=(
                    _c_from_json(ps["jones"][0]),
                    _c_from_json(ps["jones"][1]),
                ),
                basis=ps.get("basis", "HV"),
                normalize=bool(ps.get("normalize", True)),
            )

        pol_transform = None
        pt = data.get("pol_transform")
        if pt is not None:
            Jraw = pt["J"]
            J = np.array(
                [
                    [_c_from_json(Jraw[0][0]), _c_from_json(Jraw[0][1])],
                    [_c_from_json(Jraw[1][0]), _c_from_json(Jraw[1][1])],
                ],
                dtype=complex,
            )
            pol_transform = JonesMatrix(J=J, basis=pt.get("basis", "HV"))

        laser_omega0 = data.get("laser_omega0")
        if laser_omega0 is not None:
            laser_omega0 = float(laser_omega0)

        delta_omega = data.get("delta_omega", 0.0)
        if delta_omega is None:
            delta_omega = 0.0

        amplitude_kind: AmplitudeKind = data.get("amplitude_kind", "rabi")
        if amplitude_kind not in ("rabi", "field"):
            raise ValueError(f"Invalid amplitude_kind: {amplitude_kind}")
        return cls(
            envelope=env,
            omega0=float(data.get("omega0", 1.0)),
            delta_omega=float(delta_omega),
            amplitude_kind=amplitude_kind,
            label=data.get("label"),
            laser_omega0=laser_omega0,
            pol_state=pol_state,
            pol_transform=pol_transform,
        )

    def to_dict(self) -> Dict[str, Any]:
        if not isinstance(self.envelope, SerializableEnvelope):
            raise TypeError(
                "Envelope is not serializable; expected SerializableEnvelope for to_dict()."
            )

        if callable(self.delta_omega):
            raise TypeError(
                "Callable delta_omega (chirp) is not JSON serializable."
            )

        data: Dict[str, Any] = {
            "type": "classical_coherent_drive",
            "label": self.label,
            "omega0": float(self.omega0),
            "amplitude_kind": self.amplitude_kind,
            "delta_omega": float(self.delta_omega),
            "envelope": envelope_to_json(self.envelope),
            "laser_omega0": (
                None if self.laser_omega0 is None else float(self.laser_omega0)
            ),
        }

        if self.pol_state is not None:
            data["pol_state"] = {
                "basis": self.pol_state.basis,
                "normalize": bool(self.pol_state.normalize),
                "jones": [
                    _c_to_json(self.pol_state.jones[0]),
                    _c_to_json(self.pol_state.jones[1]),
                ],
            }

        if self.pol_transform is not None:
            J = self.pol_transform.J
            data["pol_transform"] = {
                "basis": self.pol_transform.basis,
                "J": [
                    [_c_to_json(J[0, 0]), _c_to_json(J[0, 1])],
                    [_c_to_json(J[1, 0]), _c_to_json(J[1, 1])],
                ],
                "truncation": {
                    "pol_dim_default": getattr(self._trunc, "pol_dim", None)
                },
            }

        return data


def ev_to_omega(E_eV: float) -> float:
    """Convert energy in eV to angular frequency in rad/s."""
    return float(E_eV) * _e / _hbar


def transition_energy_ev(EL: EnergyLevels, tr: Transition) -> float:
    """
    Map a Transition to its optical energy in eV, using your EnergyLevels fields.

    Adjust/add cases if you support more transitions (e.g. G_XX 2ph).
    """
    if tr == Transition.G_X1:
        return EL.e_G_X1[0]
    if tr == Transition.G_X2:
        return EL.e_G_X2[0]
    if tr == Transition.G_X:  # degenerate shorthand in your model
        return EL.e_G_X[0]
    if tr == Transition.X1_XX:
        return EL.e_X1_XX[0]
    if tr == Transition.X2_XX:
        return EL.e_X2_XX[0]
    if tr == Transition.X_XX:  # degenerate shorthand
        return EL.e_X_XX[0]
    if tr == Transition.G_XX:
        return EL.e_G_XX[0]
    raise ValueError(f"Unsupported transition for this helper: {tr}")


def resonant_gaussian_drive(
    *,
    EL: EnergyLevels,
    transition: Transition,
    t0: float,
    sigma: float,
    area: float = np.pi,
    omega0: float = 1.0,
    amplitude_kind: str = "rabi",  # NEW
    detuning_rad_s: float = 0.0,
    pol: JonesState | None = None,
    label: str | None = None,
) -> ClassicalCoherentDrive:

    if amplitude_kind == "rabi":
        env = GaussianEnvelope(
            t0=t0, sigma=sigma, area=area
        )  # area = pulse area (radians)
    elif amplitude_kind == "field":
        # Make envelope peak ~ 1, so omega0 is genuinely the peak E-field [V/m]
        env = GaussianEnvelope(
            t0=t0, sigma=sigma, area=sigma * np.sqrt(2.0 * np.pi)
        )
    else:
        raise ValueError(f"Unknown amplitude_kind={amplitude_kind!r}")

    wL0 = ev_to_omega(transition_energy_ev(EL, transition))

    return ClassicalCoherentDrive(
        envelope=env,
        omega0=float(omega0),
        amplitude_kind=amplitude_kind,
        laser_omega0=float(wL0),
        delta_omega=float(detuning_rad_s),
        pol_state=pol,
        label=label or f"gauss_{transition}",
    )
