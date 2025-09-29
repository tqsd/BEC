from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from qutip import Qobj
from .registry import PhotonicRegistry
from .linops import ensure_rho


@dataclass(frozen=True)
class TwoPhotonProjector:
    r"""
    Builds and applies a projector onto the subspace
    with exactly one photon in the early and exactly
    one photon in the late emission.
    """

    reg: PhotonicRegistry

    def _proj_exactly_one(self, factors: list[int]) -> Qobj:
        r"""
        Projector onto "exactly one photon within the given set of factors".

        For a set F, this is:
        .. math::
            \sum_{i \in F} P_1[i] * \prod_{j in F, j!=i} P_0[j]

        Parameters
        ----------
        factors : list[int]
            Factor indices defining the set.

        Returns
        -------
        qutip.Qobj
            Projector on the full photonic space (CSR).
        """
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
        """
        Projector onto "one photon in early AND one photon in late".

        Returns
        -------
        qutip.Qobj
            Global projector P = P_early_exactly_one * P_late_exactly_one.
        """
        P_e1 = self._proj_exactly_one(self.reg.early_factors)
        P_l1 = self._proj_exactly_one(self.reg.late_factors)
        return (P_e1 * P_l1).to("csr")

    def postselect(
        self, rho_phot: Qobj | np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Postselect a state onto the "one-and-one" two-photon subspace.

        The input is validated/normalized with `ensure_rho`. The method returns
        the normalized postselected density matrix and the success probability.

        Parameters
        ----------
        rho_phot : qutip.Qobj or numpy.ndarray
            Photonic density matrix on dims `reg.dims_phot`.

        Returns
        -------
        (numpy.ndarray, float)
            (rho_post, p2) where p2 = Tr(P rho P). If p2 <= 0, returns a
            zero matrix of shape (Dp, Dp) and 0.0.
        """
        R = ensure_rho(rho_phot, self.reg.dims_phot)
        Pi = self.projector().full()
        PiR = Pi @ R @ Pi
        p2 = float(np.trace(PiR).real)
        if p2 <= 0.0:
            Dp = self.reg.Dp
            return np.zeros((Dp, Dp), complex), 0.0
        return (PiR / p2).astype(complex), p2
