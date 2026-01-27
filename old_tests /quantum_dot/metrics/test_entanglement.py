import unittest
from types import SimpleNamespace
import numpy as np
from qutip import basis, tensor

from unittest.mock import patch

from bec.quantum_dot.metrics.entanglement import EntanglementCalculator


def ket(dims, occ):
    """|n0, n1, ...>"""
    return tensor([basis(d, n) for d, n in zip(dims, occ)])


def rho_pure(psi):
    return (psi * psi.dag()).full()


def partial_transpose_np(R, dims, late_factors):
    """
    Reference partial transpose used in tests: reshape to dims + dims,
    swap axes k <-> M+k for each k in late_factors, then reshape back.
    """
    M = len(dims)
    tens = R.reshape(tuple(dims + dims))
    for k in late_factors:
        tens = np.swapaxes(tens, k, M + k)
    return tens.reshape(R.shape)


# ---------- tests ----------


class TestEntanglementCalculator(unittest.TestCase):

    @patch("bec.quantum_dot.metrics.entanglement.ensure_rho")
    @patch("bec.quantum_dot.metrics.entanglement.partial_transpose")
    def test_bell_state_gives_logneg_1(self, pt_mock, ensure_mock):
        """
        For a two-qubit Bell state, ||PT||_1 = 2 -> log2 = 1.
        """
        dims = [2, 2]
        # |Phi+> = (|00> + |11>)/sqrt(2)
        psi = (ket(dims, [0, 0]) + ket(dims, [1, 1])).unit()
        R = rho_pure(psi)

        ensure_mock.return_value = R
        pt_mock.side_effect = lambda A, d, L: partial_transpose_np(A, d, L)

        reg = SimpleNamespace(dims_phot=dims, late_factors=[1])
        calc = EntanglementCalculator(reg)
        EN = calc.log_neg_early_late(R)

        self.assertAlmostEqual(EN, 1.0, places=12)

    @patch("bec.quantum_dot.metrics.entanglement.ensure_rho")
    @patch("bec.quantum_dot.metrics.entanglement.partial_transpose")
    def test_product_state_has_zero_logneg(self, pt_mock, ensure_mock):
        """
        Product state |00> has ||PT||_1 = 1 -> log2 = 0.
        """
        dims = [2, 2]
        psi = ket(dims, [0, 0])
        R = rho_pure(psi)

        ensure_mock.return_value = R
        pt_mock.side_effect = lambda A, d, L: partial_transpose_np(A, d, L)

        reg = SimpleNamespace(dims_phot=dims, late_factors=[1])
        calc = EntanglementCalculator(reg)
        EN = calc.log_neg_early_late(R)

        self.assertAlmostEqual(EN, 0.0, places=12)

    @patch("bec.quantum_dot.metrics.entanglement.ensure_rho")
    @patch("bec.quantum_dot.metrics.entanglement.partial_transpose")
    def test_pure_partial_entanglement_matches_formula(
        self, pt_mock, ensure_mock
    ):
        """
        |psi> = sqrt(p)|00> + sqrt(1-p)|11>
        For two qubits, ||PT||_1 = 1 + 2*sqrt(p*(1-p))
        -> E_N = log2(1 + 2*sqrt(p*(1-p)))
        """
        dims = [2, 2]
        p = 0.8
        psi = (
            np.sqrt(p) * ket(dims, [0, 0]) + np.sqrt(1 - p) * ket(dims, [1, 1])
        ).unit()
        R = rho_pure(psi)

        ensure_mock.return_value = R
        pt_mock.side_effect = lambda A, d, L: partial_transpose_np(A, d, L)

        reg = SimpleNamespace(dims_phot=dims, late_factors=[1])
        calc = EntanglementCalculator(reg)
        EN = calc.log_neg_early_late(R)

        expected = np.log2(1.0 + 2.0 * np.sqrt(p * (1 - p)))
        self.assertAlmostEqual(EN, expected, places=12)

    @patch("bec.quantum_dot.metrics.entanglement.ensure_rho")
    @patch("bec.quantum_dot.metrics.entanglement.partial_transpose")
    def test_separable_mixture_zero_logneg(self, pt_mock, ensure_mock):
        """
        rho = 0.5|00><00| + 0.5|11><11| is separable; PT has trace norm 1.
        """
        dims = [2, 2]
        rho = 0.5 * rho_pure(ket(dims, [0, 0])) + 0.5 * rho_pure(
            ket(dims, [1, 1])
        )

        ensure_mock.return_value = rho
        pt_mock.side_effect = lambda A, d, L: partial_transpose_np(A, d, L)

        reg = SimpleNamespace(dims_phot=dims, late_factors=[1])
        calc = EntanglementCalculator(reg)
        EN = calc.log_neg_early_late(rho)

        self.assertAlmostEqual(EN, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
