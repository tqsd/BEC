from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from qutip import Qobj

from .registry import PhotonicRegistry
from .linops import ensure_rho


@dataclass(frozen=True)
class PhotonCounter:
    reg: PhotonicRegistry

    def _sum_N(self, factors: list[int]) -> Qobj:
        N = None
        for f in factors:
            N = (
                self.reg.number_op_by_factor[f]
                if N is None
                else (N + self.reg.number_op_by_factor[f]).to("csr")
            )
        return N if N is not None else self.reg.I_phot * 0.0

    def counts(self, rho_phot: Qobj | np.ndarray) -> Dict[str, float]:
        R = ensure_rho(rho_phot, self.reg.dims_phot)
        Rq = Qobj(R, dims=[self.reg.dims_phot, self.reg.dims_phot]).to("csr")
        N_e = float((self._sum_N(self.reg.early_factors) * Rq).tr().real)
        N_l = float((self._sum_N(self.reg.late_factors) * Rq).tr().real)
        return {"N_early": N_e, "N_late": N_l}
