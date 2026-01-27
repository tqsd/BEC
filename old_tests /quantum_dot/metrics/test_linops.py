import unittest
import numpy as np
from qutip import basis, tensor, Qobj

from bec.quantum_dot.metrics.linops import ensure_rho, partial_transpose, purity


def ket(dims, occ):
    return tensor([basis(d, n) for d, n in zip(dims, occ)])


class TestLinops(unittest.TestCase):
    # ---------- ensure_rho ----------
    def test_ensure_rho_normalizes_numpy(self):
        dims = [2]
        # Start with |0><0| scaled by 2 -> trace 2
        rho = 2.0 * (basis(2, 0) * basis(2, 0).dag()).full()
        Rn = ensure_rho(rho, dims)
        self.assertEqual(Rn.shape, (2, 2))
        self.assertTrue(np.isclose(np.trace(Rn).real, 1.0))
        self.assertEqual(Rn.dtype, complex)

    def test_ensure_rho_accepts_qobj(self):
        dims = [2, 2]
        psi = (ket(dims, [0, 0]) + ket(dims, [1, 1])).unit()
        Rq = psi * psi.dag()  # Qobj
        Rn = ensure_rho(Rq, dims)
        self.assertEqual(Rn.shape, (4, 4))
        self.assertTrue(np.isclose(np.trace(Rn).real, 1.0))

    def test_ensure_rho_raises_on_bad_shape(self):
        dims = [2]
        rho = np.eye(3)
        with self.assertRaises(ValueError):
            _ = ensure_rho(rho, dims)

    def test_ensure_rho_raises_on_nonpositive_trace(self):
        dims = [2]
        rho = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            _ = ensure_rho(rho, dims)

    # ---------- partial_transpose ----------
    def test_partial_transpose_is_involutive(self):
        dims = [2, 2]
        # Random pure state
        rng = np.random.default_rng(42)
        vec = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi = Qobj(vec.reshape(-1, 1), dims=[[4], [1]]).unit()
        R = (psi * psi.dag()).full().reshape(4, 4)

        # PT over second subsystem (index 1), applied twice -> original
        Rpt = partial_transpose(R, dims, [1])
        Rpt2 = partial_transpose(Rpt, dims, [1])
        np.testing.assert_allclose(Rpt2, R, atol=1e-12, rtol=1e-12)

    def test_partial_transpose_of_bell_state_has_expected_spectrum(self):
        dims = [2, 2]
        # |Phi+> = (|00> + |11>)/sqrt(2)
        psi = (ket(dims, [0, 0]) + ket(dims, [1, 1])).unit()
        R = (psi * psi.dag()).full()
        Rpt = partial_transpose(R, dims, [1])
        evals = np.linalg.eigvalsh(
            0.5 * (Rpt + Rpt.conj().T)
        )  # ensure Hermitian
        # Expected eigenvalues: {0.5, 0.5, 0.5, -0.5}
        evals_sorted = np.sort(evals)
        np.testing.assert_allclose(
            evals_sorted, np.array([-0.5, 0.5, 0.5, 0.5]), atol=1e-9, rtol=0
        )

    # ---------- purity ----------
    def test_purity_pure_state_is_one(self):
        dims = [2]
        psi = basis(2, 0)
        R = (psi * psi.dag()).full()
        self.assertAlmostEqual(purity(R), 1.0, places=12)

    def test_purity_mixed_state_less_than_one(self):
        dims = [2]
        rho = (
            0.5 * (basis(2, 0) * basis(2, 0).dag()).full()
            + 0.5 * (basis(2, 1) * basis(2, 1).dag()).full()
        )
        self.assertAlmostEqual(purity(rho), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
