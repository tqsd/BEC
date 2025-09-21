from dataclasses import dataclass
from typing import List
import numpy as np
from qutip import Qobj, ket2dm, qeye, basis, tensor

from .general import GeneralKrausChannel


def vacuum_ket_for_dims(dims: List[int]) -> Qobj:
    return tensor(*[basis(d, 0) for d in dims])


@dataclass
class PrepareFromVacuum:
    rho_target: Qobj

    def build(self) -> GeneralKrausChannel:
        dims = self.rho_target.dims[0]
        N = int(np.prod(dims))
        I = qeye(N)
        vac = vacuum_ket_for_dims(dims)
        P0 = ket2dm(vac)

        evals, vecs = self.rho_target.eigenstates()
        Ks = [
            np.sqrt(float(p)) * (psi * vac.dag())
            for p, psi in zip(evals, vecs)
            if float(p) > 1e-14
        ]
        Ks.append(I - P0)

        ch = GeneralKrausChannel(Ks, dims_in=dims, dims_out=dims)
        ch.check_cptp()
        return ch


@dataclass
class PrepareFromScalar:
    rho_target: Qobj
    clip_tol: float = 1e-12

    def build(self) -> GeneralKrausChannel:
        # Ensure Hermitian
        rho = 0.5 * (self.rho_target + self.rho_target.dag())

        # Get dims
        dims_out = rho.dims[0]

        # Eigendecompose
        evals, vecs = rho.eigenstates()

        # Clip tiny negatives and renormalize to trace 1
        evals = np.array([max(0.0, float(p)) for p in evals], dtype=float)
        s = float(evals.sum())
        if s <= 0.0:
            raise ValueError(
                "rho_target has non-positive spectrum after clipping."
            )
        evals /= s

        # Build Kraus operators K_k = sqrt(p_k) |psi_k><0|
        ket0 = basis(1, 0)
        Ks = []
        for p, psi in zip(evals, vecs):
            if p > self.clip_tol:
                K = np.sqrt(p) * (psi * ket0.dag())
                K.dims = [[dims_out], [[1]]]
                Ks.append(K)

        ch = GeneralKrausChannel(Ks, dims_in=[1], dims_out=dims_out)
        ch.check_cptp()
        return ch
