from __future__ import annotations
from typing import List, Sequence
import numpy as np
from qutip import Qobj


def ensure_rho(R: Qobj | np.ndarray, dims: List[int]) -> np.ndarray:
    A = R.full() if isinstance(R, Qobj) else np.asarray(R)
    D = int(np.prod(dims))
    if A.shape != (D, D):
        raise ValueError(
            f"rho shape {A.shape} incompatible with dims {dims} (prod {D})."
        )
    tr = float(np.trace(A).real)
    if tr <= 0.0:
        raise ValueError("rho has non-positive trace.")
    return (A / tr).astype(complex)


def partial_transpose(
    R: np.ndarray, dims: List[int], sys_idxs: Sequence[int]
) -> np.ndarray:
    M = len(dims)
    tens = R.reshape(dims + dims)  # ket 0..M-1, bra M..2M-1
    for k in sys_idxs:
        tens = np.swapaxes(tens, k, k + M)
    return tens.reshape(R.shape)


def purity(R: np.ndarray) -> float:
    return float(np.real(np.trace(R @ R)))
