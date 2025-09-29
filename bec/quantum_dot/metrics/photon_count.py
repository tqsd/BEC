from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from qutip import Qobj

from .registry import PhotonicRegistry
from .linops import ensure_rho


@dataclass(frozen=True)
class PhotonCounter:
    """
    Compute expected photon numbers on early and late branches.

    The registry supplies per-factor number operators N[f] (as Qobj on the
    full photonic space), the early/late factor index lists, and the photonic
    identity. Counts are computed as:
        N_early = Tr[(sum_{f in early} N[f]) rho]
        N_late  = Tr[(sum_{f in late}  N[f]) rho]

    Parameters
    ----------
    reg : PhotonicRegistry
        Registry with fields:
          - dims_phot : list[int]
          - early_factors, late_factors : list[int]
          - number_op_by_factor : dict[int, Qobj]
          - I_phot : Qobj identity on the photonic space
    """

    reg: PhotonicRegistry

    def _sum_N(self, factors: list[int]) -> Qobj:
        """
        Sum number operators over the given factor indices.

        Parameters
        ----------
        factors : list[int]
            Photonic factor indices (0-based within the photonic tensor).

        Returns
        -------
        qutip.Qobj
            Sum of N[f] for f in factors; zero operator if the list is empty.
        """
        N = None
        for f in factors:
            N = (
                self.reg.number_op_by_factor[f]
                if N is None
                else (N + self.reg.number_op_by_factor[f]).to("csr")
            )
        return N if N is not None else self.reg.I_phot * 0.0

    def counts(self, rho_phot: Qobj | np.ndarray) -> Dict[str, float]:
        """
        Expectation values of total photon number on early/late branches.

        The input state is validated and normalized with `ensure_rho`.

        Parameters
        ----------
        rho_phot : qutip.Qobj or numpy.ndarray
            Photonic density matrix on dims `reg.dims_phot`.

        Returns
        -------
        dict[str, float]
            {"N_early": float, "N_late": float}
        """
        R = ensure_rho(rho_phot, self.reg.dims_phot)
        Rq = Qobj(R, dims=[self.reg.dims_phot, self.reg.dims_phot]).to("csr")
        N_e = float((self._sum_N(self.reg.early_factors) * Rq).tr().real)
        N_l = float((self._sum_N(self.reg.late_factors) * Rq).tr().real)
        return {"N_early": N_e, "N_late": N_l}
