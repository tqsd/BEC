from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bec.quantum_dot.enums import QDState


def _qd_basis_index(state: QDState) -> int:
    # Order must match your symbol library / qd4 basis:
    # [G, X1, X2, XX]
    if state is QDState.G:
        return 0
    if state is QDState.X1:
        return 1
    if state is QDState.X2:
        return 2
    if state is QDState.XX:
        return 3
    raise KeyError(state)


def _mixed_radix_index(indices: Sequence[int], dims: Sequence[int]) -> int:
    if len(indices) != len(dims):
        raise ValueError("indices and dims must have same length")
    out = 0
    for i, d in zip(indices, dims):
        ii = int(i)
        dd = int(d)
        if ii < 0 or ii >= dd:
            raise ValueError(f"Index {ii} out of range for dim {dd}")
        out = out * dd + ii
    return int(out)


def ket_qd_vacuum(*, dims: Sequence[int], qd_state: QDState) -> np.ndarray:
    """
    Build |psi> on full space with:
      - dot in qd_state
      - all other subsystems in |0>

    Expected dims ordering (as used everywhere else):
      (4, fock_dim, fock_dim, fock_dim, fock_dim)
    """
    dims = tuple(int(d) for d in dims)
    if not dims or dims[0] != 4:
        raise ValueError(f"Expected dims[0]==4 for QD, got dims={dims}")

    qd_i = _qd_basis_index(qd_state)

    # rest are vacuum
    local_indices = [qd_i] + [0] * (len(dims) - 1)

    D = int(np.prod(dims))
    idx = _mixed_radix_index(local_indices, dims)

    psi = np.zeros((D,), dtype=complex)
    psi[idx] = 1.0 + 0.0j
    return psi


def rho0_qd_vacuum(*, dims: Sequence[int], qd_state: QDState) -> np.ndarray:
    """
    Density matrix rho0 = |psi><psi| for dot in qd_state and fock vacua.
    """
    psi = ket_qd_vacuum(dims=dims, qd_state=qd_state)
    return np.outer(psi, np.conjugate(psi))


@dataclass(frozen=True)
class QDInitialStateFactory:
    """
    Convenience wrapper if you prefer an object.
    """

    dims: Sequence[int]

    def rho0(self, qd_state: QDState) -> np.ndarray:
        return rho0_qd_vacuum(dims=self.dims, qd_state=qd_state)

    def ket(self, qd_state: QDState) -> np.ndarray:
        return ket_qd_vacuum(dims=self.dims, qd_state=qd_state)
