from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class OverlapCalculator:
    """
    Compute Lorentzian spectral overlap and HOM visibility for
    the QD cascade.

    Inputs are decay rates (in 1/s) for the early and late emission,
    plus the splitting delta_rad_s = |FSS|/hbar in rad/s.

    The effective pari linewidth used in the overlap formula depends on
    `which`:
    - "early": 0.5 * (gamma_XX_X1 + gamma_XX_X2)
    - "late": 0.5 * (gamma_X1_G + gamma_X2_G)
    - "avg": 0.5 * ("early"+"late")

    all quantities are treadet as rates in 1/s.
    """

    # all in 1/s (angular rates acceptable—see below)
    gamma_XX_X1: float
    gamma_XX_X2: float
    gamma_X1_G: float
    gamma_X2_G: float
    delta_rad_s: float  # |FSS|/\hbar

    def _pair_linewidth(self, which: str) -> float:
        """
        Return the effective pair linewidth for the requested branch.

        Parameters
        ----------
        which : {"early", "late", "avg"}

        Returns
        -------
        float
            Effective linewidth gamma in 1/s.
        """
        if which == "early":
            g1, g2 = self.gamma_XX_X1, self.gamma_XX_X2
        elif which == "late":
            g1, g2 = self.gamma_X1_G, self.gamma_X2_G
        else:
            g1 = 0.5 * (self.gamma_X1_G + self.gamma_XX_X1)
            g2 = 0.5 * (self.gamma_X2_G + self.gamma_XX_X2)
        return 0.5 * (g1 + g2)

    def overlap(self, which: str) -> float:
        """
        Lorentzian spectral overlap for the requested branch.

        Uses: overlap = gamma / sqrt(gamma^2 + delta^2), where gamma is the
        effective pair linewidth from `_pair_linewidth(which)` and delta is
        `delta_rad_s`.

        Returns
        -------
        float
            Overlap in [0, 1]. Returns 0 if gamma <= 0.
        """
        gamma = self._pair_linewidth(which)
        if gamma <= 0.0:
            return 0.0
        return float(gamma / math.hypot(gamma, self.delta_rad_s))

    def hom(self, which: str) -> float:
        """
        HOM visibility under Lorentzian envelopes for the requested branch.

        Returns
        -------
        float
            Visibility in [0, 1], equal to overlap(which)**2.
        """
        lam = self.overlap(which)
        return float(lam * lam)
