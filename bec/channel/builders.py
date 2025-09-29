from dataclasses import dataclass
from typing import List
import numpy as np
from qutip import Qobj, ket2dm, qeye, basis, tensor

from .general import GeneralKrausChannel


def vacuum_ket_for_dims(dims: List[int]) -> Qobj:
    """
    Return the vacuum ket for a tensor product with factor dims.

    Parameters
    ----------
    dims: List[int]
        Local dimensions of each factor

    Returns
    -------
    qutip.Qobj
        Ket with dims and amplitude 1 at all-zero index.
    """
    return tensor(*[basis(d, 0) for d in dims])


@dataclass
class PrepareFromVacuum:
    """
    Build a CPTP channel that prepares `rho_target` starting from vacuum.

    Given `rho_target`, the channel maps |vac><vac| -> rho_target.

    Attributes:
    -----------
    rho_target: qutip.Qobj
        Target density martix (dims `[dims, dims]`)

    Notes
    -----
    Kraus operators are:
      - `sqrt(p_k) |psi_k><vac|` for eigenpairs of `rho_target` with p_k > 1e-14
      - `I-|vac><vac|` to complete trace preservation
    """

    rho_target: Qobj

    def build(self) -> GeneralKrausChannel:
        """
        Construct and return the generalized Kraus channel.

        Returns
        -------
        GeneralKrausChannel
            Channel with ``dims_in = dims_out = rho_target.dims[0]``.

        Raises
        ------
        ValueError
            If ``rho_target`` is incompatible with vacuum dims.
        """
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
    """
    Build a CPTP channel that prepares ``rho_target`` from a 1D input space.

    The channel embeds a scalar input (dim_in=[1]) and outputs a state on
    dims_out = rho_target.dims[0]. The target is Hermitized and its spectrum
    is clipped at 0 with optional tolerance, then renormalized to trace 1
    before forming Kraus operators.

    Attributes
    ----------
    rho_target : qutip.Qobj
        Target density operator (not necessarily perfectly valid; it will be
        symmetrized).
    clip_tol : float
        Threshold below which eigenvalues are ignored when forming Kraus ops.

    Raises
    ------
    ValueError
        If all eigenvalues are clipped so the spectrum is non-positive.
    """

    rho_target: Qobj
    clip_tol: float = 1e-12

    def build(self) -> GeneralKrausChannel:
        """
        Construct and return the generalized Kraus channel.

        Returns
        -------
        GeneralKrausChannel
            Channel with ``dims_in = [1]`` and ``dims_out = rho_target.dims[0]``.

        Raises
        ------
        ValueError
            If the clipped spectrum sums to zero (non-positive).
        """
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
