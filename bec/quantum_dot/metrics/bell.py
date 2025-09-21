from qutip import Qobj, basis, tensor
import numpy as np
from dataclasses import dataclass
from .registry import PhotonicRegistry
from .linops import ensure_rho


@dataclass(frozen=True)
class BellAnalyzer:
    reg: PhotonicRegistry

    # ---------- helpers ----------

    def _all_pm_indices(self):
        """Return lists of factor indices for early/late and +/- across ALL branches."""
        off = self.reg.offset
        ef = sorted(self.reg.early_factors)
        lf = sorted(self.reg.late_factors)
        e_plus = [f for f in ef if ((f - off) % 2) == 0]
        e_minus = [f for f in ef if ((f - off) % 2) == 1]
        l_plus = [f for f in lf if ((f - off) % 2) == 0]
        l_minus = [f for f in lf if ((f - off) % 2) == 1]
        return e_plus, e_minus, l_plus, l_minus

    def _ket_occ(self, occ_01):
        kets = [basis(d, n) for d, n in zip(self.reg.dims_phot, occ_01)]
        return tensor(kets).to("csr")

    def _ket_two_ones(self, i: int, j: int) -> Qobj:
        occ = [0] * len(self.reg.dims_phot)
        occ[i] = 1
        occ[j] = 1
        return self._ket_occ(occ)

    def _superposition_ket(self, first_idxs, second_idxs) -> Qobj:
        """Equal-weight superposition over all |1_i, 1_j> with i in first_idxs, j in second_idxs."""
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
        """Return aggregated basis kets: |e+,l+>, |e+,l->, |e-,l+>, |e-,l->."""
        e_p, e_m, l_p, l_m = self._all_pm_indices()
        ket_pp = self._superposition_ket(e_p, l_p)
        ket_pm = self._superposition_ket(e_p, l_m)
        ket_mp = self._superposition_ket(e_m, l_p)
        ket_mm = self._superposition_ket(e_m, l_m)
        return ket_pp, ket_pm, ket_mp, ket_mm

    # ---------- stable numeric core (numpy inner products) ----------

    @staticmethod
    def _as_np(x: Qobj) -> np.ndarray:
        """Dense numpy array, shape (D,1) or (D,D)."""
        a = x.full() if isinstance(x, Qobj) else np.asarray(x)
        return np.asarray(a)

    def _diag_elem_np(self, Rq: Qobj, ket: Qobj) -> float:
        """<ket| R |ket> computed via numpy (robust to dims/types)."""
        if ket.norm() == 0.0:
            return 0.0
        psi = self._as_np(ket)  # (D,1)
        R = self._as_np(Rq)  # (D,D)
        val = psi.conj().T @ R @ psi  # (1,1)
        return float(np.real(val[0, 0]))

    def _offdiag_np(self, Rq: Qobj, bra: Qobj, ket: Qobj) -> complex:
        """<bra| R |ket> via numpy."""
        if bra.norm() == 0.0 or ket.norm() == 0.0:
            return 0.0 + 0.0j
        phi = self._as_np(bra)
        psi = self._as_np(ket)
        R = self._as_np(Rq)
        val = phi.conj().T @ R @ psi
        return complex(val[0, 0])

    # ---------- main API ----------

    def analyze(self, rho_2ph):
        """
        Analyze the post-selected two-photon state on polarization
        (aggregating spectral branches per polarization).
        Returns weights, cross-coherence (abs/phase), and Bell fidelity bound.
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
