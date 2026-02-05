from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bec.metrics.linops import _dims_prod, partial_trace


@dataclass(frozen=True)
class PhotonDecomposition:
    p0: float
    p1_total: float
    p2_exact: float


def photon_number_distribution(
    rho_modes: np.ndarray, mode_dims: Sequence[int], nmax: int
) -> np.ndarray:
    """
    Return p[n] for n=0..nmax where n is total photon number across all modes.
    Assumes rho_modes is in the product Fock basis for those modes.
    """
    rho_modes = np.asarray(rho_modes, dtype=complex)
    mode_dims = [int(d) for d in mode_dims]
    Dm = _dims_prod(mode_dims)
    if rho_modes.shape != (Dm, Dm):
        raise ValueError("rho_modes shape mismatch")

    diag = np.real_if_close(np.diag(rho_modes)).astype(float)
    # enumerate basis states by unravel index
    p = np.zeros(int(nmax) + 1, dtype=float)

    for flat, prob in enumerate(diag):
        occ = np.unravel_index(flat, mode_dims)
        n = int(sum(int(x) for x in occ))
        if n <= nmax:
            p[n] += float(prob)

    return p


def decompose_photons(
    rho_full: np.ndarray, dims: Sequence[int], keep_mode_indices: Sequence[int]
) -> PhotonDecomposition:
    """
    Decompose photon number for a selected group of mode indices.
    Traces out everything else (including QD and other modes), then computes:
      p0, p1_total, p2_exact
    """
    dims = [int(d) for d in dims]
    keep = [int(k) for k in keep_mode_indices]

    rho_keep = partial_trace(rho_full, dims, keep=keep)
    mode_dims = [dims[i] for i in keep]

    p = photon_number_distribution(rho_keep, mode_dims, nmax=2)
    return PhotonDecomposition(
        p0=float(p[0]),
        p1_total=float(p[1]),
        p2_exact=float(p[2]),
    )
