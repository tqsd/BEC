import unittest
from types import SimpleNamespace
import numpy as np
from qutip import basis, tensor

from bec.quantum_dot.metrics.bell import BellAnalyzer

from unittest.mock import patch


def ket_two_ones(dims, i, j):
    """Helper: |...1_i...1_j...> on the photonic space."""
    occ = [0] * len(dims)
    occ[i] = 1
    occ[j] = 1
    return tensor([basis(d, n) for d, n in zip(dims, occ)]).to("csr")


class TestBellAnalyzer(unittest.TestCase):
    def test_all_pm_indices_no_offset(self):
        # early: factors 0,1 ; late: factors 2,3 ; offset = 0
        reg = SimpleNamespace(
            dims_phot=[2, 2, 2, 2],
            early_factors=[0, 1],
            late_factors=[2, 3],
            offset=0,
        )
        BA = BellAnalyzer(reg)
        e_p, e_m, l_p, l_m = BA._all_pm_indices()
        # With offset=0, plus are even, minus are odd
        self.assertEqual(e_p, [0])
        self.assertEqual(e_m, [1])
        self.assertEqual(l_p, [2])
        self.assertEqual(l_m, [3])

    def test_all_pm_indices_with_offset(self):
        # early: 1,2 ; late: 3,4 ; offset = 1
        # Now (idx - offset) % 2 == 0 => plus are 1 and 3; minus are 2 and 4
        reg = SimpleNamespace(
            dims_phot=[2, 2, 2, 2, 2],
            early_factors=[1, 2],
            late_factors=[3, 4],
            offset=1,
        )
        BA = BellAnalyzer(reg)
        e_p, e_m, l_p, l_m = BA._all_pm_indices()
        self.assertEqual(e_p, [1])
        self.assertEqual(e_m, [2])
        self.assertEqual(l_p, [3])
        self.assertEqual(l_m, [4])

    @patch("bec.quantum_dot.metrics.bell.ensure_rho")
    def test_analyze_bell_cross_state(self, ensure_rho_mock):
        """
        Build a state |psi> = (|e+,l-> + e^{iφ} |e-,l+>) / sqrt(2),
        with indices (e+, e-, l+, l-) = (0,1,2,3). Expect:
          p_pm = 1/2, p_mp = 1/2, p_pp = p_mm = 0
          |coherence| = 1/2, phase = -φ
          F_max = 1.0
        """
        dims = [2, 2, 2, 2]
        e_plus, e_minus, l_plus, l_minus = 0, 1, 2, 3
        ket_pm = ket_two_ones(dims, e_plus, l_minus)
        ket_mp = ket_two_ones(dims, e_minus, l_plus)

        phi = 0.7  # arbitrary phase
        psi = (ket_pm + (np.cos(phi) + 1j * np.sin(phi)) * ket_mp).unit()
        R = (psi * psi.dag()).full()  # numpy array OK for ensure_rho return
        ensure_rho_mock.return_value = R

        reg = SimpleNamespace(
            dims_phot=dims,
            early_factors=[e_plus, e_minus],
            late_factors=[l_plus, l_minus],
            offset=0,
        )
        BA = BellAnalyzer(reg)
        out = BA.analyze(R)

        # weights
        w = out["weights"]
        self.assertAlmostEqual(w["p_pp"], 0.0, places=12)
        self.assertAlmostEqual(w["p_mm"], 0.0, places=12)
        self.assertAlmostEqual(w["p_pm"], 0.5, places=12)
        self.assertAlmostEqual(w["p_mp"], 0.5, places=12)
        self.assertAlmostEqual(w["parallel"], 0.0, places=12)
        self.assertAlmostEqual(w["cross"], 1.0, places=12)

        # coherence magnitude and phase (phase is -phi modulo 2π)
        coh = out["coherence_cross"]
        self.assertAlmostEqual(coh["abs"], 0.5, places=12)

        # Compare angle modulo 2π
        def angle_close(a, b, tol=1e-9):
            # wrap difference into [-pi, pi]
            diff = (a - b + np.pi) % (2 * np.pi) - np.pi
            return abs(diff) < tol

        self.assertTrue(
            angle_close(coh["phase_rad"], -phi),
            msg=f"phase {coh['phase_rad']} not equal to {-phi} (mod 2π)",
        )

        # Bell fidelity bound
        self.assertAlmostEqual(out["bell_fidelity_max"], 1.0, places=12)

    @patch("bec.quantum_dot.metrics.bell.ensure_rho")
    def test_analyze_with_empty_index_sets_returns_zeros(self, ensure_rho_mock):
        """
        If early/late sets are empty, the superposition kets are zero.
        Then all weights and coherence should be zero; bound = 0.
        """
        dims = [2, 2]  # any non-empty photonic space is fine
        D = np.prod(dims)
        # use maximally mixed state as a benign input (ensure_rho returns normalized)
        ensure_rho_mock.return_value = np.eye(int(D)) / float(D)

        reg = SimpleNamespace(
            dims_phot=dims,
            early_factors=[],
            late_factors=[],
            offset=0,
        )
        BA = BellAnalyzer(reg)
        out = BA.analyze(ensure_rho_mock.return_value)

        w = out["weights"]
        for key in ("p_pp", "p_pm", "p_mp", "p_mm", "parallel", "cross"):
            self.assertAlmostEqual(w[key], 0.0, places=12)

        coh = out["coherence_cross"]
        self.assertAlmostEqual(coh["abs"], 0.0, places=12)
        # phase is undefined; implementation returns angle(0) = 0
        self.assertAlmostEqual(out["bell_fidelity_max"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
