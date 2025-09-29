from __future__ import annotations
from dataclasses import dataclass
from qutip import Qobj
import numpy as np
from .registry import PhotonicRegistry
from .linops import ensure_rho


@dataclass(frozen=True)
class PopulationDecomposer:
    """
    Decompose a photonic state into vacuum (p0), single-photon total (p1_total),
    exact two-photon (p2_exact), and a residual multiphoton probability.

    The projectors are constructed from the registry as:
      - P0: product over factors of the per-factor vacuum projector P0[f]
      - P1_total: sum over factors f of P1[f] times the product of P0[j] for all j != f
      - P2: provided externally as an argument to `p0_p1_p2_exact_multi`

    The multiphoton remainder is computed as:
        p_multiphoton = max(0, 1 - (p0 + p1_total + p2_exact))

    Parameters
    ----------
    reg : PhotonicRegistry
        Registry providing:
          - dims_phot: list of local photonic dimensions
          - I_phot: identity operator on the photonic space (Qobj)
          - proj0_by_factor: dict[int, Qobj] of per-factor vacuum projectors,
            each extended to the full photonic space
          - proj1_by_factor: dict[int, Qobj] of per-factor one-photon projectors,
            each extended to the full photonic space
    """

    reg: PhotonicRegistry

    def _Pi0_all(self) -> Qobj:
        """
        Projector onto the global photonic vacuum.

        Returns
        -------
        qutip.Qobj
            P0 = prod_f P0[f], as a CSR operator with dims [dims_phot, dims_phot].
        """
        P = self.reg.I_phot
        for f in self.reg.proj0_by_factor.keys():
            P = (P * self.reg.proj0_by_factor[f]).to("csr")
        return P

    def _Pi1_total(self) -> Qobj:
        r"""
        Projector onto the subspace with exactly one photon across all factors.

        Constructed as:

        .. math::
            \sum_f P_1[f] * \prod_{j != f} P_0[j]

        Returns
        -------
        qutip.Qobj
            P1_total as a CSR operator with dims [dims_phot, dims_phot].
        """
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
        r"""
        Decompose a photonic density matrix into p0, p1_total, p2_exact, and
        residual multiphoton.

        The input state is validated and normalized on the photonic space
        using `ensure_rho`.
        Probabilities are computed as traces of the corresponding projectors:
        .. math::
            p_0 = \text{Tr}[P_0 R], p_{1,total} = \text{Tr}[P_{1,total} R], p_{2,exact} = \text{Tr}[P_2 R]

        The remainder
            p_multiphoton = max(0, 1 - (p0 + p1_total + p2_exact))
        captures probability outside these subspaces.

        Parameters
        ----------
        rho_phot : qutip.Qobj or numpy.ndarray
            Photonic density matrix (possibly unnormalized); validated by
            `ensure_rho`.
        P2 : qutip.Qobj
            Projector onto the exact two-photon subspace of interest, defined
            on the same photonic space.

        Returns
        -------
        dict
            {
              "p0": float,
              "p1_total": float,
              "p2_exact": float,
              "p_multiphoton": float
            }
        """
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
