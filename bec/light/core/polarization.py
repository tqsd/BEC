from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


HV_BASIS = "HV"


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


def _require_hv(basis: Any) -> None:
    if basis is None:
        return
    if str(basis) != HV_BASIS:
        raise ValueError(
            f"Only basis={HV_BASIS!r} is supported (got {basis!r})"
        )


@dataclass(frozen=True)
class JonesState:
    """
    Jones polarization state in the fixed HV basis.

    Stored as two complex amplitudes (E_H, E_V).
    """

    jones: Tuple[complex, complex] = (1.0 + 0j, 0.0 + 0j)
    normalize: bool = True
    basis: str = HV_BASIS  # fixed, validated

    def __post_init__(self) -> None:
        _require_hv(self.basis)

        j0 = complex(self.jones[0])
        j1 = complex(self.jones[1])
        object.__setattr__(self, "jones", (j0, j1))
        object.__setattr__(self, "basis", HV_BASIS)

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
            "basis": HV_BASIS,
            "normalize": bool(self.normalize),
            "jones": [_c_to_json(self.jones[0]), _c_to_json(self.jones[1])],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JonesState":
        _require_hv(data.get("basis", HV_BASIS))
        j = data["jones"]
        return cls(
            jones=(_c_from_json(j[0]), _c_from_json(j[1])),
            normalize=bool(data.get("normalize", True)),
            basis=HV_BASIS,
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
    2x2 complex Jones matrix in the fixed HV basis.
    """

    J: np.ndarray
    basis: str = HV_BASIS

    def __post_init__(self) -> None:
        _require_hv(self.basis)

        A = np.asarray(self.J, dtype=np.complex128)
        if A.shape != (2, 2):
            raise ValueError("JonesMatrix must be 2x2 complex")
        object.__setattr__(self, "J", A)
        object.__setattr__(self, "basis", HV_BASIS)

    def apply(self, state: JonesState) -> np.ndarray:
        # state.basis is always HV, but keep the check as a guardrail.
        if state.basis != HV_BASIS:
            raise ValueError("Jones basis mismatch")
        return self.J @ state.as_array()

    def to_dict(self) -> Dict[str, Any]:
        J = self.J
        return {
            "type": "jones_matrix",
            "basis": HV_BASIS,
            "J": [
                [_c_to_json(J[0, 0]), _c_to_json(J[0, 1])],
                [_c_to_json(J[1, 0]), _c_to_json(J[1, 1])],
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JonesMatrix":
        _require_hv(data.get("basis", HV_BASIS))
        Jraw = data["J"]
        J = np.array(
            [
                [_c_from_json(Jraw[0][0]), _c_from_json(Jraw[0][1])],
                [_c_from_json(Jraw[1][0]), _c_from_json(Jraw[1][1])],
            ],
            dtype=np.complex128,
        )
        return cls(J=J, basis=HV_BASIS)

    @classmethod
    def identity(cls) -> "JonesMatrix":
        return cls(J=np.eye(2, dtype=np.complex128))

    @classmethod
    def rotation(cls, theta_rad: float) -> "JonesMatrix":
        c = float(np.cos(theta_rad))
        s = float(np.sin(theta_rad))
        J = np.array([[c, -s], [s, c]], dtype=np.complex128)
        return cls(J=J)

    @classmethod
    def retarder(
        cls, delta_rad: float, theta_rad: float = 0.0
    ) -> "JonesMatrix":
        Rm = cls.rotation(-theta_rad).J
        Rp = cls.rotation(theta_rad).J
        D = np.array(
            [
                [np.exp(-1j * delta_rad / 2.0), 0.0],
                [0.0, np.exp(+1j * delta_rad / 2.0)],
            ],
            dtype=np.complex128,
        )
        return cls(J=Rm @ D @ Rp)

    @classmethod
    def hwp(cls, theta_rad: float = 0.0) -> "JonesMatrix":
        return cls.retarder(np.pi, theta_rad)

    @classmethod
    def qwp(cls, theta_rad: float = 0.0) -> "JonesMatrix":
        return cls.retarder(np.pi / 2.0, theta_rad)

    @classmethod
    def linear_polarizer(cls, theta_rad: float = 0.0) -> "JonesMatrix":
        Rm = cls.rotation(-theta_rad).J
        Rp = cls.rotation(theta_rad).J
        P = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        return cls(J=Rm @ P @ Rp)


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
    return pol_transform.J @ E
