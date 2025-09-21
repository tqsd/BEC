from __future__ import annotations
from dataclasses import dataclass
from qutip import Qobj
import numpy as np
from .registry import PhotonicRegistry
from .linops import ensure_rho


@dataclass(frozen=True)
class PopulationDecomposer:
    reg: PhotonicRegistry

    def _Pi0_all(self) -> Qobj:
        P = self.reg.I_phot
        for f in self.reg.proj0_by_factor.keys():
            P = (P * self.reg.proj0_by_factor[f]).to("csr")
        return P

    def _Pi1_total(self) -> Qobj:
        # Σ_f P1[f] Π_{j≠f} P0[j]
        out = self.reg.I_phot * 0.0
        for f in self.reg.proj1_by_factor.keys():
            term = self.reg.proj1_by_factor[f]
            for j in self.reg.proj0_by_factor.keys():
                if j == f:
                    continue
                term = (term * self.reg.proj0_by_factor[j]).to("csr")
            out = (out + term).to("csr")
        return out

    def p0_p1_p2_exact_multi(
        self, rho_phot: Qobj | np.ndarray, P2: Qobj
    ) -> dict:
        R = ensure_rho(rho_phot, self.reg.dims_phot)
        Rq = Qobj(R, dims=[self.reg.dims_phot, self.reg.dims_phot]).to("csr")

        P0 = self._Pi0_all()
        P1 = self._Pi1_total()

        p0 = float((P0 * Rq).tr().real)
        p1 = float((P1 * Rq).tr().real)
        p2 = float((P2 * Rq).tr().real)

        # Whatever is left is "multiphoton" (>= 2 photons but not exactly 1+1 across early/late)
        pm = max(0.0, 1.0 - (p0 + p1 + p2))
        return {"p0": p0, "p1_total": p1, "p2_exact": p2, "p_multiphoton": pm}
