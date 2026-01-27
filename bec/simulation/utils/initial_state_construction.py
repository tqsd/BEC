import numpy as np
from typing import List

from bec.operators.qd_operators import QDState


def ket0_dm(d: int) -> np.ndarray:
    rho = np.zeros((d, d), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def qd_ground_dm() -> np.ndarray:
    # QD has dim 4, and you define QDState.G = 0
    rho = np.zeros((4, 4), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for A in mats[1:]:
        out = np.kron(out, A)
    return out


def qd_basis_ket(state: QDState) -> np.ndarray:
    v = np.zeros((4,), dtype=np.complex128)
    v[int(state)] = 1.0
    return v


def dm_from_ket(ket: np.ndarray) -> np.ndarray:
    ket = np.asarray(ket, dtype=np.complex128).reshape(-1)
    return np.outer(ket, ket.conj())


def default_rho0_from_dims(
    dims: List[int],
    *,
    qd_state: QDState = QDState.G,
) -> np.ndarray:
    """
    Initial state: QD in |qd_state><qd_state|, all other subsystems vacuum.
    """
    if not dims or dims[0] != 4:
        raise ValueError(f"Expected dims[0]=4 for QD, got dims={dims}")

    qd_dm = dm_from_ket(qd_basis_ket(qd_state))
    parts = [qd_dm] + [ket0_dm(int(d)) for d in dims[1:]]
    return kron_all(parts)
