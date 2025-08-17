import numpy as np
from enum import Enum, auto
from dataclasses import dataclass


class Pol(Enum):
    PLUS = auto()
    MINUS = auto()


class Ladder(Enum):
    A = auto()
    A_DAG = auto()
    N = auto()
    I = auto()


def ladder_operator(
    dim: int, operator: Ladder = Ladder.A, normalized: bool = False
) -> np.ndarray:
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        if normalized:
            val = 1
        else:
            val = np.sqrt(n)
        if operator == Ladder.A:
            a[n - 1, n] = val
        elif operator == Ladder.A_DAG:
            a[n, n - 1] = val
        else:
            raise ValueError("operator must be 'annihilation' or 'creation'")
    return a


def rotated_ladder_operator(
    dim: int,
    theta: float,
    phi: float = 0.0,
    pol: Pol = Pol.PLUS,
    operator: Ladder = Ladder.A,
    normalized: bool = False,
) -> np.ndarray:
    """Orthogonal (unitary) mix of H,V modes on H⊗V space."""
    if operator not in [Ladder.A, Ladder.A_DAG]:
        raise ValueError("Wrong operator, only A and A_DAG accepted")
    a = ladder_operator(dim, operator=Ladder.A, normalized=normalized)
    I = np.eye(dim, dtype=complex)
    aH, aV = np.kron(a, I), np.kron(I, a)

    # SU(2) mixing
    c, s, eip, eim = (
        np.cos(theta),
        np.sin(theta),
        np.exp(1j * phi),
        np.exp(-1j * phi),
    )
    if pol == Pol.PLUS:
        A = c * aH + eip * s * aV
    elif pol == Pol.MINUS:
        A = -eim * s * aH + c * aV
    else:
        raise ValueError("mode must be 'plus' or 'minus'")

    return A if operator == Ladder.A else A.conj().T


def _vac_projector(dim):
    P0 = np.zeros((dim, dim), dtype=complex)
    P0[0, 0] = 1.0
    return P0


def vacuum_projector(dim: int):
    P0_plus = _vac_projector(dim)
    P0_minus = _vac_projector(dim)
    return np.kron(P0_plus, P0_minus)
