import unittest
import numpy as np
from qutip import basis, tensor, qeye, Qobj

from bec.quantum_dot.metrics.photon_count import PhotonCounter


def number_projector_for_factor(dims_phot, f_idx):
    """Qobj projector onto |1> on factor f_idx, identity elsewhere (photonic-only)."""
    ops = []
    for i, d in enumerate(dims_phot):
        if i == f_idx:
            ket1 = basis(d, 1)
            ops.append((ket1 * ket1.dag()).to("csr"))
        else:
            ops.append(qeye(d).to("csr"))
    return tensor(ops).to("csr")


class FakePhotonicRegistry:
    """Minimal registry for PhotonCounter tests."""

    def __init__(self, dims_phot, early_factors, late_factors):
        self.dims_phot = list(dims_phot)
        self.early_factors = list(early_factors)
        self.late_factors = list(late_factors)

        Dp = int(np.prod(dims_phot))
        self.I_phot = Qobj(np.eye(Dp), dims=[dims_phot, dims_phot]).to("csr")

        # Build N[f] = |1><1| on factor f (identity elsewhere)
        self.number_op_by_factor = {
            f: number_projector_for_factor(dims_phot, f)
            for f in range(len(dims_phot))
        }


class PhotonCounterTests(unittest.TestCase):
    def setUp(self):
        # Two modes (+,-) each -> 4 photonic factors, all truncated to {0,1}
        self.dims = [2, 2, 2, 2]
        self.early = [0, 1]  # mode 0 (+,-)
        self.late = [2, 3]  # mode 1 (+,-)
        self.reg = FakePhotonicRegistry(self.dims, self.early, self.late)
        self.pc = PhotonCounter(self.reg)

    def _rho_basis(self, occ):
        """Density matrix for |n0, n1, n2, n3>."""
        ket = tensor([basis(d, n) for d, n in zip(self.dims, occ)]).to("csr")
        return (ket * ket.dag()).full()

    def test_sum_N_empty_returns_zero(self):
        Z = self.pc._sum_N([])
        self.assertIsInstance(Z, Qobj)
        # Expectation on any state is zero; check on vacuum
        rho0 = self._rho_basis([0, 0, 0, 0])
        val = float(
            (Z * Qobj(rho0, dims=[self.dims, self.dims]).to("csr")).tr().real
        )
        self.assertAlmostEqual(val, 0.0, places=12)

    def test_counts_on_vacuum_is_zero(self):
        rho0 = self._rho_basis([0, 0, 0, 0])
        out = self.pc.counts(rho0)
        self.assertAlmostEqual(out["N_early"], 0.0, places=12)
        self.assertAlmostEqual(out["N_late"], 0.0, places=12)

    def test_counts_on_single_photon_basis_states(self):
        # One photon on factor 0 (early +)
        rho = self._rho_basis([1, 0, 0, 0])
        out = self.pc.counts(rho)
        self.assertAlmostEqual(out["N_early"], 1.0, places=12)
        self.assertAlmostEqual(out["N_late"], 0.0, places=12)

        # One photon on factor 3 (late -)
        rho = self._rho_basis([0, 0, 0, 1])
        out = self.pc.counts(rho)
        self.assertAlmostEqual(out["N_early"], 0.0, places=12)
        self.assertAlmostEqual(out["N_late"], 1.0, places=12)

    def test_counts_on_superposition_splits_probability(self):
        # |psi> = (|1,0,0,0> + |0,0,0,1>) / sqrt(2)
        ket_a = tensor(
            [basis(2, 1), basis(2, 0), basis(2, 0), basis(2, 0)]
        ).unit()
        ket_b = tensor(
            [basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 1)]
        ).unit()
        psi = (ket_a + ket_b).unit()
        rho = (psi * psi.dag()).full()

        out = self.pc.counts(rho)
        # number operators are diagonal in Fock basis -> each branch has 0.5 expectation
        self.assertAlmostEqual(out["N_early"], 0.5, places=12)
        self.assertAlmostEqual(out["N_late"], 0.5, places=12)

    def test_counts_accepts_qobj_inputs(self):
        rho = Qobj(
            self._rho_basis([1, 0, 0, 0]), dims=[self.dims, self.dims]
        ).to("csr")
        out = self.pc.counts(rho)
        self.assertAlmostEqual(out["N_early"], 1.0, places=12)
        self.assertAlmostEqual(out["N_late"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
