from __future__ import annotations

import numpy as np

from bec.metrics.linops import partial_transpose


def log_negativity(
    rho: np.ndarray, dims: tuple[int, int], sys: int = 0
) -> float:
    """
    Logarithmic negativity for a 2-part system with dims (dA, dB).
    sys: which side to partial transpose (0 or 1)

    Returns log2(||rho^T_sys||_1).
    """
    rho = np.asarray(rho, dtype=complex)
    dA, dB = int(dims[0]), int(dims[1])
    if rho.shape != (dA * dB, dA * dB):
        raise ValueError("rho shape mismatch for dims")

    dims_full = (dA, dB)
    rho_pt = partial_transpose(rho, dims_full, sys=int(sys))
    s = np.linalg.svd(rho_pt, compute_uv=False)
    tr_norm = float(np.sum(np.abs(s)))
    if tr_norm <= 0.0:
        return 0.0
    return float(np.log2(tr_norm))
