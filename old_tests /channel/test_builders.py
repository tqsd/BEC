import unittest
from unittest.mock import patch
import numpy as np
from qutip import basis, ket2dm, qeye, Qobj

from bec.channel.builders import (
    PrepareFromScalar,
    PrepareFromVacuum,
    vacuum_ket_for_dims,
)


class EchoChannel:
    def __init__(self, Ks, dims_in, dims_out):
        self.Ks = Ks
        self.dims_in = list(dims_in)
        self.dims_out = list(dims_out)
        self.cptp_checked = False

    def check_cptp(self):
        self.cptp_checked = True


class BuildersTests(unittest.TestCase):
    def test_vacuum_ket_for_dims_shape_and_state(self):
        dims = [2, 3]
        vac = vacuum_ket_for_dims(dims)
        # Shape is prod(dims) x 1
        self.assertEqual(vac.shape, (6, 1))
        # The amplitude at the all-zero basis is 1, others 0
        arr = vac.full().ravel()
        self.assertTrue(np.isclose(arr[0], 1.0))
        self.assertTrue(np.allclose(arr[1:], 0.0))

    @patch("bec.channel.builders.GeneralKrausChannel", EchoChannel)
    def test_prepare_from_vacuum_builds_expected_kraus(self):
        # Target: pure |1><1| on a qubit
        psi1 = basis(2, 1)
        rho_target = ket2dm(psi1)  # dims [[2],[2]]
        builder = PrepareFromVacuum(rho_target)

        ch = builder.build()
        self.assertIsInstance(ch, EchoChannel)
        self.assertEqual(ch.dims_in, [2])
        self.assertEqual(ch.dims_out, [2])
        self.assertTrue(ch.cptp_checked)

        # Expect two Kraus ops:
        # K0 = |1><0|  (since p=1 for |1>)
        # K1 = I - |0><0|
        vac = basis(2, 0)
        K0_expected = (psi1 * vac.dag()).full()
        P0 = ket2dm(vac).full()
        K1_expected = (qeye(2) - Qobj(P0)).full()

        self.assertEqual(len(ch.Ks), 2)
        np.testing.assert_allclose(ch.Ks[0].full(), K0_expected, atol=1e-12)
        np.testing.assert_allclose(ch.Ks[1].full(), K1_expected, atol=1e-12)

    @patch("bec.channel.builders.GeneralKrausChannel", EchoChannel)
    def test_prepare_from_scalar_hermitize_clip_and_error(self):
        # Negative definite -> after clipping, sum of evals is zero -> error
        rho_bad = -1.0 * qeye(2)  # dims [[2],[2]]
        builder = PrepareFromScalar(rho_bad)
        with self.assertRaises(ValueError):
            _ = builder.build()

        # Slightly non-Hermitian target is Hermitized; still diagonalizable
        A = ket2dm(basis(2, 0)) + 1e-9j * (basis(2, 0) * basis(2, 1).dag())
        builder2 = PrepareFromScalar(A)
        ch2 = builder2.build()
        self.assertTrue(ch2.cptp_checked)


if __name__ == "__main__":
    unittest.main()
