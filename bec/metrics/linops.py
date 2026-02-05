from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _as_list_int(xs: Iterable[int]) -> list[int]:
    return [int(x) for x in xs]


def _dims_prod(dims: Sequence[int]) -> int:
    out = 1
    for d in dims:
        out *= int(d)
    return int(out)


def partial_trace(
    rho: np.ndarray, dims: Sequence[int], keep: Sequence[int]
) -> np.ndarray:
    """
    Partial trace over all subsystems not in keep.

    rho: (D, D) density matrix
    dims: local dimensions, length N
    keep: indices of subsystems to keep (0..N-1)

    Returns rho_keep with shape (D_keep, D_keep).
    """
    rho = np.asarray(rho, dtype=complex)
    dims = [int(d) for d in dims]
    n = len(dims)

    keep = sorted({int(k) for k in keep})
    if any((k < 0 or k >= n) for k in keep):
        raise ValueError("keep indices out of range")

    D = _dims_prod(dims)
    if rho.shape != (D, D):
        raise ValueError(f"rho has shape {rho.shape}, expected {(D, D)}")

    traced = [i for i in range(n) if i not in keep]

    # reshape into 2N tensor: (i0..iN-1, j0..jN-1)
    reshaped = rho.reshape(tuple(dims) + tuple(dims))

    # trace over each traced subsystem: axis i_k with axis j_k
    for k in reversed(traced):
        n_curr = reshaped.ndim // 2
        reshaped = np.trace(reshaped, axis1=k, axis2=k + n_curr)

    d_keep = [dims[i] for i in keep]
    D_keep = _dims_prod(d_keep)
    return reshaped.reshape((D_keep, D_keep))


def partial_transpose(
    rho: np.ndarray, dims: Sequence[int], sys: int
) -> np.ndarray:
    """
    Partial transpose on subsystem 'sys' (0..N-1) for a bipartite-like test.
    Works for general multipartite dims.

    rho: (D, D)
    dims: local dims
    sys: which subsystem to transpose
    """
    rho = np.asarray(rho, dtype=complex)
    dims = [int(d) for d in dims]
    n = len(dims)
    sys = int(sys)

    D = _dims_prod(dims)
    if rho.shape != (D, D):
        raise ValueError(f"rho has shape {rho.shape}, expected {(D, D)}")
    if sys < 0 or sys >= n:
        raise ValueError("sys out of range")

    t = rho.reshape(tuple(dims) + tuple(dims))
    # swap i_sys and j_sys
    axes = list(range(2 * n))
    axes[sys], axes[sys + n] = axes[sys + n], axes[sys]
    t_pt = np.transpose(t, axes=axes)
    return t_pt.reshape((D, D))
