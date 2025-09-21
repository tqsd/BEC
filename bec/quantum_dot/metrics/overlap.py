from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class OverlapCalculator:
    # all in 1/s (angular rates acceptable—see below)
    gamma_XX_X1: float
    gamma_XX_X2: float
    gamma_X1_G: float
    gamma_X2_G: float
    delta_rad_s: float  # |FSS|/ħ

    def _pair_linewidth(self, which: str) -> float:
        if which == "early":
            g1, g2 = self.gamma_XX_X1, self.gamma_XX_X2
        elif which == "late":
            g1, g2 = self.gamma_X1_G, self.gamma_X2_G
        else:
            g1 = 0.5 * (self.gamma_X1_G + self.gamma_XX_X1)
            g2 = 0.5 * (self.gamma_X2_G + self.gamma_XX_X2)
        return 0.5 * (g1 + g2)

    def overlap(self, which: str) -> float:
        gamma = self._pair_linewidth(which)
        if gamma <= 0.0:
            return 0.0
        return float(gamma / math.hypot(gamma, self.delta_rad_s))

    def hom(self, which: str) -> float:
        lam = self.overlap(which)
        return float(lam * lam)
