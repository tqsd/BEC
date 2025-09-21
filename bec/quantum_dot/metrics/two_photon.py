from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from qutip import Qobj
from .registry import PhotonicRegistry
from .linops import ensure_rho


@dataclass(frozen=True)
class TwoPhotonProjector:
    reg: PhotonicRegistry

    def _proj_exactly_one(self, factors: list[int]) -> Qobj:
        I = self.reg.I_phot
        terms = []
        for i in factors:
            Pi1 = self.reg.proj1_by_factor[i]  # |1><1| on factor i
            P0_others = I
            for j in factors:
                if j == i:
                    continue
                P0_others = (P0_others * self.reg.proj0_by_factor[j]).to("csr")
            terms.append((Pi1 * P0_others).to("csr"))
        out = I * 0.0
        for T in terms:
            out = (out + T).to("csr")
        return out

    def projector(self) -> Qobj:
        P_e1 = self._proj_exactly_one(self.reg.early_factors)
        P_l1 = self._proj_exactly_one(self.reg.late_factors)
        return (P_e1 * P_l1).to("csr")

    def postselect(
        self, rho_phot: Qobj | np.ndarray
    ) -> tuple[np.ndarray, float]:
        R = ensure_rho(rho_phot, self.reg.dims_phot)
        Pi = self.projector().full()
        PiR = Pi @ R @ Pi
        p2 = float(np.trace(PiR).real)
        if p2 <= 0.0:
            Dp = self.reg.Dp
            return np.zeros((Dp, Dp), complex), 0.0
        return (PiR / p2).astype(complex), p2
