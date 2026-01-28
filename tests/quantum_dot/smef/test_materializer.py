import unittest

import numpy as np

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.materialize import materialize_op_expr

from bec.quantum_dot.enums import QDState, Transition
from bec.quantum_dot.smef.materializer import default_qd_materializer
from bec.quantum_dot.smef.symbols import qd_basis_index


class TestQDMaterializer(unittest.TestCase):
    def test_resolve_local_qd_symbol(self) -> None:
        ctx = default_qd_materializer(register_fock_dims=(2,))
        T = ctx.resolve_symbol(Transition.G_XX, (4,))
        self.assertEqual(T.shape, (4, 4))

        iG = qd_basis_index(QDState.G)
        iXX = qd_basis_index(QDState.XX)
        self.assertAlmostEqual(float(np.abs(T[iXX, iG])), 1.0)

    def test_resolve_local_mode_symbol(self) -> None:
        ctx = default_qd_materializer(register_fock_dims=(2,))
        a = ctx.resolve_symbol("a", (2,))
        self.assertEqual(a.shape, (2, 2))
        self.assertAlmostEqual(float(np.abs(a[0, 1])), 1.0)

    def test_materialize_embedded_kron(self) -> None:
        # dims: (qd, m1, m2) = (4,2,2)
        dims = (4, 2, 2)
        ctx = default_qd_materializer(register_fock_dims=(2,))

        qd = 0
        m1 = 1

        # Operator: |XX><G| on qd and adag on m1
        expr = OpExpr.atom(
            EmbeddedKron(
                indices=(qd, m1),
                locals=(LocalSymbolOp(Transition.G_XX), LocalSymbolOp("adag")),
            )
        )
        M = materialize_op_expr(expr, dims=dims, ctx=ctx)
        self.assertEqual(M.shape, (16, 16))

        # Check one matrix element: (XX,1,0) <- (G,0,0)
        # Flatten index for basis ordering kron(qd,m1,m2).
        iG = qd_basis_index(QDState.G)
        iXX = qd_basis_index(QDState.XX)

        # basis index: ((qd * 2) + m1) * 2 + m2
        col = ((iG * 2) + 0) * 2 + 0
        row = ((iXX * 2) + 1) * 2 + 0

        self.assertAlmostEqual(float(np.abs(M[row, col])), 1.0)

    def test_materialize_drive_like_term(self) -> None:
        # sx_G_XX embedded on qd only, identity on modes
        dims = (4, 2, 2)
        ctx = default_qd_materializer(register_fock_dims=(2,))

        expr = OpExpr.atom(
            EmbeddedKron(indices=(0,), locals=(LocalSymbolOp("sx_G_XX"),))
        )
        M = materialize_op_expr(expr, dims=dims, ctx=ctx)
        self.assertEqual(M.shape, (16, 16))

        # Should couple |G,0,0> <-> |XX,0,0>
        iG = qd_basis_index(QDState.G)
        iXX = qd_basis_index(QDState.XX)
        idx_G00 = ((iG * 2) + 0) * 2 + 0
        idx_XX00 = ((iXX * 2) + 0) * 2 + 0

        self.assertAlmostEqual(float(np.abs(M[idx_XX00, idx_G00])), 1.0)
        self.assertAlmostEqual(float(np.abs(M[idx_G00, idx_XX00])), 1.0)


if __name__ == "__main__":
    unittest.main()
