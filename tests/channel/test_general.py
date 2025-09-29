import unittest
import numpy as np
from qutip import Qobj, basis, ket2dm, qeye

from bec.channel.general import GeneralKrausChannel


class TestGeneralKrausChannel(unittest.TestCase):
    def test_apply_identity_channel(self):
        # ρ = |+><+|, K = I
        ket_plus = (basis(2, 0) + basis(2, 1)).unit()
        rho = ket2dm(ket_plus)
        K = qeye(2)
        ch = GeneralKrausChannel([K], dims_in=[2], dims_out=[2])

        # TP check passes
        ch.check_cptp()

        # Applying identity channel returns the same state
        rho_out = ch.apply(rho)
        np.testing.assert_allclose(
            rho_out.full(), rho.full(), atol=1e-12, rtol=0.0
        )

        # as_super exists and is square with shape (N^2, N^2) = (4, 4)
        S = ch.as_super()
        self.assertIsInstance(S, Qobj)
        self.assertEqual(S.shape, (4, 4))

    def test_check_cptp_failure(self):
        # Single Kraus K = sqrt(1/2) * I -> not trace-preserving (sum K^†K = 0.5 I)
        K = (1.0 / np.sqrt(2.0)) * qeye(2)
        ch = GeneralKrausChannel([K], dims_in=[2], dims_out=[2])
        with self.assertRaises(ValueError):
            ch.check_cptp()

    def test_dephasing_channel_via_projectors(self):
        # Kraus set {P0, P1} yields complete dephasing in computational basis
        ket0, ket1 = basis(2, 0), basis(2, 1)
        P0 = ket2dm(ket0)
        P1 = ket2dm(ket1)
        ch = GeneralKrausChannel([P0, P1], dims_in=[2], dims_out=[2])

        # TP check passes (P0 + P1 = I)
        ch.check_cptp()

        # Input |+><+| should map to diag(1/2, 1/2)
        ket_plus = (ket0 + ket1).unit()
        rho_in = ket2dm(ket_plus)
        rho_out = ch.apply(rho_in)

        expected = 0.5 * ket2dm(ket0) + 0.5 * ket2dm(ket1)
        np.testing.assert_allclose(
            rho_out.full(), expected.full(), atol=1e-12, rtol=0.0
        )

    def test_as_super_non_square_raises(self):
        # Rectangular Kraus (3x2) => dims_in=[2], dims_out=[3]
        K = Qobj(np.zeros((3, 2), dtype=complex))
        K.dims = [[3], [2]]
        ch = GeneralKrausChannel([K], dims_in=[2], dims_out=[3])
        with self.assertRaises(ValueError):
            _ = ch.as_super()

    def test_apply_rectangular_kraus_shapes_correctly(self):
        # Simple isometry from qubit (2) to qutrit (3):
        # K maps |0> -> |0>, |1> -> |1>, ignores |2>
        K = Qobj(np.array([[1, 0], [0, 1], [0, 0]], dtype=complex))
        K.dims = [[3], [2]]

        ch = GeneralKrausChannel([K], dims_in=[2], dims_out=[3])

        # Input pure |+><+|
        ket_plus = (basis(2, 0) + basis(2, 1)).unit()
        rho_in = ket2dm(ket_plus)

        rho_out = ch.apply(rho_in)
        self.assertIsInstance(rho_out, Qobj)
        self.assertEqual(rho_out.shape, (3, 3))
        # Expect the state embedded in the first two levels of the qutrit
        expected_block = ket2dm((basis(2, 0) + basis(2, 1)).unit()).full()
        full_expected = np.zeros((3, 3), dtype=complex)
        full_expected[:2, :2] = expected_block
        np.testing.assert_allclose(
            rho_out.full(), full_expected, atol=1e-12, rtol=0.0
        )

    def test_check_cptp_respects_dims_in_out(self):
        # Build a valid TP channel with two rectangular Kraus operators that form an isometry:
        # K0 = |0_out><0_in|, K1 = |1_out><1_in|
        K0 = Qobj(np.array([[1, 0], [0, 0], [0, 0]], dtype=complex))
        K1 = Qobj(np.array([[0, 0], [0, 1], [0, 0]], dtype=complex))
        for K in (K0, K1):
            K.dims = [[3], [2]]

        ch = GeneralKrausChannel([K0, K1], dims_in=[2], dims_out=[3])
        # sum K^† K = I_2 -> TP on input space of dimension 2
        ch.check_cptp()  # should not raise


if __name__ == "__main__":
    unittest.main()
