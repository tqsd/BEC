from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np


PolBasis = Literal["HV"]


def _c_to_json(z: complex) -> list[float]:
    return [float(np.real(z)), float(np.imag(z))]


def _c_from_json(x: Any) -> complex:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return complex(float(x), 0.0)
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return complex(float(x[0]), float(x[1]))
    if isinstance(x, dict) and "re" in x and "im" in x:
        return complex(float(x["re"]), float(x["im"]))
    raise TypeError(f"Cannot decode complex from {type(x)}: {x!r}")


@dataclass(frozen=True)
class JonesState:
    """
    Jones polarization state in a 2D basis (default HV).

    Stored as two complex amplitudes (Ex, Ey) in the given basis.
    """

    jones: Tuple[complex, complex] = (1.0 + 0j, 0.0 + 0j)
    basis: PolBasis = "HV"
    normalize: bool = True

    def as_array(self) -> np.ndarray:
        v = np.array(self.jones, dtype=np.complex128)
        if self.normalize:
            n = np.linalg.norm(v)
            if n != 0.0:
                v = v / n
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "jones_state",
            "basis": self.basis,
            "normalize": bool(self.normalize),
            "jones": [_c_to_json(self.jones[0]), _c_to_json(self.jones[1])],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JonesState":
        j = data["jones"]
        return cls(
            jones=(_c_from_json(j[0]), _c_from_json(j[1])),
            basis=data.get("basis", "HV"),
            normalize=bool(data.get("normalize", True)),
        )

    @classmethod
    def H(cls) -> "JonesState":
        return cls(jones=(1.0 + 0j, 0.0 + 0j))

    @classmethod
    def V(cls) -> "JonesState":
        return cls(jones=(0.0 + 0j, 1.0 + 0j))

    @classmethod
    def D(cls) -> "JonesState":
        return cls(jones=(1.0 + 0j, 1.0 + 0j))

    @classmethod
    def A(cls) -> "JonesState":
        return cls(jones=(1.0 + 0j, -1.0 + 0j))

    @classmethod
    def R(cls) -> "JonesState":
        s = 1.0 / np.sqrt(2.0)
        return cls(jones=(s + 0j, -1j * s))

    @classmethod
    def L(cls) -> "JonesState":
        s = 1.0 / np.sqrt(2.0)
        return cls(jones=(s + 0j, 1j * s))


@dataclass(frozen=True)
class JonesMatrix:
    """
    2x2 complex Jones matrix in a basis (default HV).
    """

    J: np.ndarray
    basis: PolBasis = "HV"

    def __post_init__(self) -> None:
        A = np.asarray(self.J, dtype=np.complex128)
        if A.shape != (2, 2):
            raise ValueError("JonesMatrix must be 2x2 complex")
        object.__setattr__(self, "J", A)

    def apply(self, state: JonesState) -> np.ndarray:
        if state.basis != self.basis:
            raise ValueError("Jones basis mismatch")
        return self.J @ state.as_array()

    def to_dict(self) -> Dict[str, Any]:
        J = self.J
        return {
            "type": "jones_matrix",
            "basis": self.basis,
            "J": [
                [_c_to_json(J[0, 0]), _c_to_json(J[0, 1])],
                [_c_to_json(J[1, 0]), _c_to_json(J[1, 1])],
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JonesMatrix":
        Jraw = data["J"]
        J = np.array(
            [
                [_c_from_json(Jraw[0][0]), _c_from_json(Jraw[0][1])],
                [_c_from_json(Jraw[1][0]), _c_from_json(Jraw[1][1])],
            ],
            dtype=np.complex128,
        )
        return cls(J=J, basis=data.get("basis", "HV"))

    @classmethod
    def identity(cls, *, basis: PolBasis = "HV") -> "JonesMatrix":
        return cls(J=np.eye(2, dtype=np.complex128), basis=basis)

    @classmethod
    def rotation(
        cls, theta_rad: float, *, basis: PolBasis = "HV"
    ) -> "JonesMatrix":
        c = float(np.cos(theta_rad))
        s = float(np.sin(theta_rad))
        J = np.array([[c, -s], [s, c]], dtype=np.complex128)
        return cls(J=J, basis=basis)

    @classmethod
    def retarder(
        cls, delta_rad: float, theta_rad: float = 0.0, *, basis: PolBasis = "HV"
    ) -> "JonesMatrix":
        Rm = cls.rotation(-theta_rad, basis=basis).J
        Rp = cls.rotation(theta_rad, basis=basis).J
        D = np.array(
            [
                [np.exp(-1j * delta_rad / 2.0), 0.0],
                [0.0, np.exp(+1j * delta_rad / 2.0)],
            ],
            dtype=np.complex128,
        )
        return cls(J=Rm @ D @ Rp, basis=basis)

    @classmethod
    def hwp(
        cls, theta_rad: float = 0.0, *, basis: PolBasis = "HV"
    ) -> "JonesMatrix":
        return cls.retarder(np.pi, theta_rad, basis=basis)

    @classmethod
    def qwp(
        cls, theta_rad: float = 0.0, *, basis: PolBasis = "HV"
    ) -> "JonesMatrix":
        return cls.retarder(np.pi / 2.0, theta_rad, basis=basis)

    @classmethod
    def linear_polarizer(
        cls, theta_rad: float = 0.0, *, basis: PolBasis = "HV"
    ) -> "JonesMatrix":
        Rm = cls.rotation(-theta_rad, basis=basis).J
        Rp = cls.rotation(theta_rad, basis=basis).J
        P = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        return cls(J=Rm @ P @ Rp, basis=basis)


def effective_polarization(
    *,
    pol_state: Optional[JonesState],
    pol_transform: Optional[JonesMatrix],
) -> Optional[np.ndarray]:
    """
    Return the effective polarization vector after applying an optional transform.
    Result is a length-2 complex numpy array, normalized if pol_state.normalize is True.
    """
    if pol_state is None:
        return None
    E = pol_state.as_array()
    if pol_transform is None:
        return E
    if pol_transform.basis != pol_state.basis:
        raise ValueError("Polarization basis mismatch")
    return pol_transform.J @ E
