from __future__ import annotations
from typing import List, Sequence
import numpy as np
from qutip import Qobj


def ensure_rho(R: Qobj | np.ndarray, dims: List[int]) -> np.ndarray:
    r"""
    Validates the shape and normalizes a density matrix on a composite
    Hilbert space.

    The input is converted to a dense numpy array, checked to have
    shape (D,D) with D=prod(dims), and scaled to unit trace.the result
    is returned as a complex ndarray.

    Parameters
    ----------
    R: qutip.Qobj or numpy.ndarray
        Candidate density matrix
    dims: list[int]
        Local dimensions of the composite space.their product must
        mach the matrix size.

    Returns
    -------
    numpy.ndarray
        Normalized density matrix (trace=1) complex type
    """
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
    """
    Partial transpose over the subsystems listed in `sys_idxs`.

    The matrix is reshaped to rang-2M tensor with ket axes 0..M-1
    and bra axes M...2M-1, then for each k in `sys_idxs` the pair
    of axes (k,M+k) is swapped. The result is reshaped back to (D,D).

    Parameters
    ----------
    R: numpy.ndarrray
        Density matrix with shape (D,D) where D=prod(dims)
    dims: list[int]
        Local dimensions of the composite space
    sys_idxs: list[int]
        Indices of subsystems to transpose (0-based over the ket factors)

    Returns
    -------
    numpy.ndarray
        Partially transposed matrix with the same shape as `R`

    """
    M = len(dims)
    tens = R.reshape(dims + dims)  # ket 0..M-1, bra M..2M-1
    for k in sys_idxs:
        tens = np.swapaxes(tens, k, k + M)
    return tens.reshape(R.shape)


def purity(R: np.ndarray) -> float:
    """
    Compute Tr(R^2) for a density matrix.

    Parameters
    ----------
    R: numpy.ndarray
        Density matrix (assumed normalized)

    Returns
    -------
    float
        Purity in [0,1] for valid states.
        - 1  -> pure state
        - <1 -> mixed state
    """
    return float(np.real(np.trace(R @ R)))
