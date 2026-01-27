import unittest
import numpy as np
from unittest.mock import patch
from qutip import Qobj

from bec.quantum_dot.me.base_builder import BaseBuilder


LEVELS = ["G", "X1", "X2", "XX"]
IDX = {name: i for i, name in enumerate(LEVELS)}


def basis_ket(i: int, N: int = 4) -> np.ndarray:
    v = np.zeros((N, 1), dtype=complex)
    v[i, 0] = 1.0
    return v


def op_matrix(bra: str, ket: str) -> np.ndarray:
    i = IDX[bra]
    j = IDX[ket]
    return basis_ket(i) @ basis_ket(j).conj().T


class FakeKronPad:
    def pad(self, local_op, subsystem_label, idx):
        # dot-only embedding
        return local_op


def make_fake_ctx():
    ctx = {}
    for bra in LEVELS:
        for ket in LEVELS:
            mat = op_matrix(bra, ket)
            ctx[f"s_{bra}_{ket}"] = lambda _ignored, m=mat: m
    return ctx


class TestBaseBuilder(unittest.TestCase):
    def setUp(self):
        self.ctx = make_fake_ctx()
        self.kron = FakeKronPad()
        self.bb = BaseBuilder(context=self.ctx, kron=self.kron)

    def test_require_ctx_missing_key_raises(self):
        with self.assertRaises(KeyError):
            self.bb._require_ctx("not_a_key")

    def test_op_builds_qobj_with_dims(self):
        dims = [4]
        with patch(
            "bec.quantum_dot.me.base_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            O = self.bb.op("X1", "G", dims)

        self.assertIsInstance(O, Qobj)
        self.assertEqual(O.dims, [dims, dims])

    def test_op_matrix_correctness(self):
        dims = [4]
        with patch(
            "bec.quantum_dot.me.base_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            O = self.bb.op("X2", "XX", dims)

        expected = Qobj(op_matrix("X2", "XX"), dims=[dims, dims]).to("csr")
        self.assertLess((O - expected).norm(), 1e-12)


if __name__ == "__main__":
    unittest.main()
