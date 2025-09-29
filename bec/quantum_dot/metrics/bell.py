from qutip import Qobj, basis, tensor
import numpy as np
from dataclasses import dataclass
from .registry import PhotonicRegistry
from .linops import ensure_rho


@dataclass(frozen=True)
class BellAnalyzer:
    r"""
    Polarization Bell analysis on a post-selected two-photon state.

    This analyzer aggregates spectral emissions into four polarization basis
    kets:
    - |e+, l+>, |e+,l->, |e-,l+>, |e-,l->

    Using the photonic registry, it builds the kets, extracts diagonal
    weights on the cross/parallel subspaces, computes the cross coherence
    between |e+,l-> and |e-,l+> and reports a simple Bell fidelity upper
    bound
    .. math::
        F_{max} = 0.5 (p_\pm + p_\mp) +|\langle e_+, l_- | R | e_-, l_+\rangle |

    Parameters:
    -----------
    reg: PhotonicRegistry
        Registry dsecribing the photonic Hilbert space
        (dims_phot, early_factors, late_factors, offset)
    """

    reg: PhotonicRegistry

    # ---------- helpers ----------

    def _all_pm_indices(self):
        """
        Compute early/late and +/- factor index lists.

        Returns
        -------
        tuple[list[int], list[int], list[int], list[int]]
            (early_plus, early_minus, late_plus, late_minus), with indices
            measured in the photonic register including `offset`.
        """
        off = self.reg.offset
        ef = sorted(self.reg.early_factors)
        lf = sorted(self.reg.late_factors)
        e_plus = [f for f in ef if ((f - off) % 2) == 0]
        e_minus = [f for f in ef if ((f - off) % 2) == 1]
        l_plus = [f for f in lf if ((f - off) % 2) == 0]
        l_minus = [f for f in lf if ((f - off) % 2) == 1]
        return e_plus, e_minus, l_plus, l_minus

    def _ket_occ(self, occ_01):
        """
        Build a tensor-product ket with 0/1 occupation per photonic factor.

        Parameters
        ----------
        occ_01 : sequence[int]
            Length equals len(reg.dims_phot); each entry 0 or 1.

        Returns
        -------
        qutip.Qobj
            Ket on the photonic space with dims `[dims_phot]`.
        """
        kets = [basis(d, n) for d, n in zip(self.reg.dims_phot, occ_01)]
        return tensor(kets).to("csr")

    def _ket_two_ones(self, i: int, j: int) -> Qobj:
        """
        Ket with exactly one photon in factors i and j (others in vacuum).

        Parameters
        ----------
        i, j : int
            Photonic factor indices.

        Returns
        -------
        qutip.Qobj
            |...1_i...1_j...> as a CSR ket.
        """
        occ = [0] * len(self.reg.dims_phot)
        occ[i] = 1
        occ[j] = 1
        return self._ket_occ(occ)

    def _superposition_ket(self, first_idxs, second_idxs) -> Qobj:
        """
        Equal-weight superposition of |1_i, 1_j> over i in first_idxs, j in
        second_idxs.

        If the index sets are empty, returns the zero ket on the correct space.

        Parameters
        ----------
        first_idxs : iterable[int]
        second_idxs : iterable[int]

        Returns
        -------
        qutip.Qobj
            Unit-normalized superposition ket (CSR).
        """
        kets = [
            self._ket_two_ones(i, j) for i in first_idxs for j in second_idxs
        ]
        if not kets:
            # zero ket with correct dims
            return (
                tensor([basis(d, 0) for d in self.reg.dims_phot]).to("csr")
                * 0.0
            )
        psi = sum(kets)
        return psi.unit()

    def _basis_vectors_aggregated(self):
        """
        Aggregated polarization basis kets.

        Returns
        -------
        tuple[qutip.Qobj, qutip.Qobj, qutip.Qobj, qutip.Qobj]
            (|e+,l+>, |e+,l->, |e-,l+>, |e-,l->) as CSR kets.
        """
        e_p, e_m, l_p, l_m = self._all_pm_indices()
        ket_pp = self._superposition_ket(e_p, l_p)
        ket_pm = self._superposition_ket(e_p, l_m)
        ket_mp = self._superposition_ket(e_m, l_p)
        ket_mm = self._superposition_ket(e_m, l_m)
        return ket_pp, ket_pm, ket_mp, ket_mm

    @staticmethod
    def _as_np(x: Qobj) -> np.ndarray:
        """
        Convert Qobj or array-like to a dense numpy array.

        Returns
        -------
        numpy.ndarray
            Shape (D, 1) for kets or (D, D) for operators.
        """
        a = x.full() if isinstance(x, Qobj) else np.asarray(x)
        return np.asarray(a)

    def _diag_elem_np(self, Rq: Qobj, ket: Qobj) -> float:
        """
        Compute <ket| R |ket> using numpy, robust to dims/types.

        Returns 0.0 if `ket` has zero norm.

        Parameters
        ----------
        Rq : qutip.Qobj
            Density matrix on the photonic space.
        ket : qutip.Qobj
            State ket on the same space.

        Returns
        -------
        float
            Real expectation value.
        """
        if ket.norm() == 0.0:
            return 0.0
        psi = self._as_np(ket)  # (D,1)
        R = self._as_np(Rq)  # (D,D)
        val = psi.conj().T @ R @ psi  # (1,1)
        return float(np.real(val[0, 0]))

    def _offdiag_np(self, Rq: Qobj, bra: Qobj, ket: Qobj) -> complex:
        """
        Compute <bra| R |ket> using numpy.

        Returns 0 if either vector has zero norm.

        Parameters
        ----------
        Rq : qutip.Qobj
            Density matrix on the photonic space.
        bra : qutip.Qobj
            Bra vector (as a ket object).
        ket : qutip.Qobj
            Ket vector.

        Returns
        -------
        complex
            Complex matrix element.
        """
        if bra.norm() == 0.0 or ket.norm() == 0.0:
            return 0.0 + 0.0j
        phi = self._as_np(bra)
        psi = self._as_np(ket)
        R = self._as_np(Rq)
        val = phi.conj().T @ R @ psi
        return complex(val[0, 0])

    def analyze(self, rho_2ph):
        """
        Analyze a post-selected two-photon polarization state.

        The input is normalized (or renormalized) to the photonic subspace,
        the four aggregated basis kets are built, and the following metrics
        are returned:
          - weights: p_pp, p_pm, p_mp, p_mm, and totals "parallel" (pp+mm)
            and "cross" (pm+mp)
          - coherence_cross: absolute value and phase (rad/deg) of
            <e+,l-| R |e-,l+>
          - bell_fidelity_max: F_max = 0.5*(p_pm + p_mp) + |coherence|

        Parameters
        ----------
        rho_2ph : qutip.Qobj or array-like
            Two-photon density matrix on the photonic space, or any array
            broadcastable to that space. It is validated and normalized via
            `ensure_rho(dims=self.reg.dims_phot)`.

        Returns
        -------
        dict
            {
              "weights": {
                  "p_pp", "p_pm", "p_mp", "p_mm",
                  "parallel", "cross"
              },
              "coherence_cross": {
                  "abs", "phase_rad", "phase_deg"
              },
              "bell_fidelity_max": float
            }
        """
        R = ensure_rho(rho_2ph, self.reg.dims_phot)
        Rq = Qobj(R, dims=[self.reg.dims_phot, self.reg.dims_phot]).to("csr")

        ket_pp, ket_pm, ket_mp, ket_mm = self._basis_vectors_aggregated()

        # Diagonals
        p_pp = self._diag_elem_np(Rq, ket_pp)
        p_pm = self._diag_elem_np(Rq, ket_pm)
        p_mp = self._diag_elem_np(Rq, ket_mp)
        p_mm = self._diag_elem_np(Rq, ket_mm)

        # Cross coherence between |e+,l-> and |e-,l+>
        c = self._offdiag_np(Rq, ket_pm, ket_mp)
        coh_abs = float(abs(c))
        phase = float(np.angle(c))
        phase_deg = float(np.degrees(phase))

        # Simple Bell fidelity upper bound on the cross subspace
        F_max = float(0.5 * (p_pm + p_mp) + coh_abs)

        return {
            "weights": {
                "p_pp": p_pp,
                "p_pm": p_pm,
                "p_mp": p_mp,
                "p_mm": p_mm,
                "parallel": p_pp + p_mm,
                "cross": p_pm + p_mp,
            },
            "coherence_cross": {
                "abs": coh_abs,
                "phase_rad": phase,
                "phase_deg": phase_deg,
            },
            "bell_fidelity_max": F_max,
        }
