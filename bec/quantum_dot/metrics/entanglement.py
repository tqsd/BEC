from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .registry import PhotonicRegistry
from .linops import ensure_rho, partial_transpose


@dataclass(frozen=True)
class EntanglementCalculator:
    reg: PhotonicRegistry

    def log_neg_early_late(self, rho_phot: np.ndarray) -> float:
        R = ensure_rho(rho_phot, self.reg.dims_phot)
        Rpt = partial_transpose(R, self.reg.dims_phot, self.reg.late_factors)
        s = np.linalg.svd(Rpt, compute_uv=False)
        return float(np.log2(np.sum(s).real))
