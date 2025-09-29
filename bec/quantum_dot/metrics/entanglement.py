from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .registry import PhotonicRegistry
from .linops import ensure_rho, partial_transpose


@dataclass(frozen=True)
class EntanglementCalculator:
    r"""
    Compute bipartite entanglement (logarithmic negativity) across the early
    late emissions.

    This class takes a photonic registry (dims and factor indexing) and
    computes, for a given density matrix rho, the quantity:
    .. math::
        E_N = \log_2(||\rho^{T_{late}} ||_1)

    T_late denotes partial transpose over the late factors, and
    ||.||_1 is the trace norm (sum of singular values).

    Parameters:
    -----------
    reg: PhotonicRegistry
        Registry describing the photonic space and the bipartition
    """

    reg: PhotonicRegistry

    def log_neg_early_late(self, rho_phot: np.ndarray) -> float:
        """
        Returns the logarithmic negativity across (early)|(late).

        Steps:
        1. Validate/normalize input with ensure_rho
        2. Take partial transpose over late factors
        3. Compute trace norm via SVD and return log2

        Parameters:
        -----------
        rho_phot: np.ndarray

        Returns:
        --------
        float
           Logarithmic negativity
        """
        R = ensure_rho(rho_phot, self.reg.dims_phot)
        Rpt = partial_transpose(R, self.reg.dims_phot, self.reg.late_factors)
        s = np.linalg.svd(Rpt, compute_uv=False)
        return float(np.log2(np.sum(s).real))
