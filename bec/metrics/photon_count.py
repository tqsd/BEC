from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def number_operator_local(d: int) -> np.ndarray:
    d = int(d)
    return np.diag(np.arange(d, dtype=float)).astype(complex)


def embed_local_op(
    op: np.ndarray, dims: Sequence[int], which: int
) -> np.ndarray:
    """
    Embed a local operator acting on subsystem 'which' into full space.
    """
    dims = [int(x) for x in dims]
    which = int(which)
    n = len(dims)

    if which < 0 or which >= n:
        raise ValueError("which out of range")
    if op.shape != (dims[which], dims[which]):
        raise ValueError("op has wrong shape for subsystem")

    out = None
    for i in range(n):
        if i == which:
            a = op
        else:
            a = np.eye(dims[i], dtype=complex)
        out = a if out is None else np.kron(out, a)
    return np.asarray(out, dtype=complex)


def expect_op(rho: np.ndarray, op: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=complex)
    op = np.asarray(op, dtype=complex)
    return float(np.real_if_close(np.trace(rho @ op)))


def expected_n_per_mode(
    rho: np.ndarray, dims: Sequence[int], mode_index: int
) -> float:
    op = embed_local_op(
        number_operator_local(int(dims[int(mode_index)])), dims, int(mode_index)
    )
    return expect_op(rho, op)


def expected_n_group(
    rho: np.ndarray, dims: Sequence[int], indices: Sequence[int]
) -> float:
    total = 0.0
    for idx in indices:
        total += expected_n_per_mode(rho, dims, int(idx))
    return float(total)
