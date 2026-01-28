import unittest

import numpy as np

from bec.quantum_dot.enums import QDState, Transition
from bec.quantum_dot.smef.symbols import (
    build_default_symbol_library,
    fock_ops,
    qd4_projector,
    qd4_transition_op,
    qd_basis_index,
)


class TestQDSymbols(unittest.TestCase):
    def test_basis_order(self) -> None:
        self.assertEqual(qd_basis_index(QDState.G), 0)
        self.assertEqual(qd_basis_index(QDState.X1), 1)
        self.assertEqual(qd_basis_index(QDState.X2), 2)
        self.assertEqual(qd_basis_index(QDState.XX), 3)

    def test_projectors(self) -> None:
        P = qd4_projector(QDState.XX)
        self.assertEqual(P.shape, (4, 4))
        self.assertAlmostEqual(float(np.real(P[3, 3])), 1.0)
        self.assertAlmostEqual(float(np.real(np.sum(P))), 1.0)

    def test_transition_direction(self) -> None:
        # Transition.G_XX means G -> XX, operator is |XX><G|
        T = qd4_transition_op(Transition.G_XX)
        self.assertEqual(T.shape, (4, 4))

        iG = qd_basis_index(QDState.G)
        iXX = qd_basis_index(QDState.XX)

        self.assertAlmostEqual(float(np.abs(T[iXX, iG])), 1.0)
        self.assertAlmostEqual(float(np.abs(T[iG, iXX])), 0.0)

    def test_fock_ops_dim2(self) -> None:
        ops = fock_ops(2)
        a = ops["a"]
        adag = ops["adag"]
        n = ops["n"]
        I = ops["I"]

        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(adag.shape, (2, 2))
        self.assertEqual(n.shape, (2, 2))
        self.assertEqual(I.shape, (2, 2))

        # dim=2: a = |0><1|
        self.assertAlmostEqual(float(np.abs(a[0, 1])), 1.0)
        self.assertAlmostEqual(float(np.abs(a[1, 0])), 0.0)

        # n = diag(0,1)
        self.assertAlmostEqual(float(np.real(n[0, 0])), 0.0)
        self.assertAlmostEqual(float(np.real(n[1, 1])), 1.0)

    def test_library_resolve(self) -> None:
        lib = build_default_symbol_library(register_fock_dims=(2,))
        # QD symbol by Transition enum
        T = lib.resolve(Transition.X1_XX, (4,))
        self.assertEqual(T.shape, (4, 4))

        # QD legacy symbol
        P = lib.resolve("proj_G", (4,))
        self.assertEqual(P.shape, (4, 4))

        # mode symbol
        adag = lib.resolve("adag", (2,))
        self.assertEqual(adag.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
