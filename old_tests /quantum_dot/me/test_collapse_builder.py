import unittest
import numpy as np
from unittest.mock import patch
from qutip import Qobj

from bec.quantum_dot.me.collapse_builder import CollapseBuilder
from bec.quantum_dot.me.types import CollapseTermKind


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
        return local_op


def make_fake_ctx():
    ctx = {}
    for bra in LEVELS:
        for ket in LEVELS:
            mat = op_matrix(bra, ket)
            ctx[f"s_{bra}_{ket}"] = lambda _ignored, m=mat: m
    return ctx


class TestCollapseBuilder(unittest.TestCase):
    def setUp(self):
        self.ctx = make_fake_ctx()
        self.kron = FakeKronPad()
        self.cb = CollapseBuilder(
            context=self.ctx, kron=self.kron, pm_map=None)

    def test_catalog_counts(self):
        dims = [4]
        with patch(
            "bec.quantum_dot.me.base_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            terms = self.cb.build_catalog(dims)

        # radiative: 4, phonon dephasing: 3 => total 7
        self.assertEqual(len(terms), 7)
        self.assertEqual(
            sum(t.kind == CollapseTermKind.RADIATIVE for t in terms), 4
        )
        self.assertEqual(
            sum(t.kind == CollapseTermKind.PHONON for t in terms), 3
        )

    def test_radiative_ops_match_transitions(self):
        dims = [4]
        with patch(
            "bec.quantum_dot.me.base_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            terms = self.cb.build_catalog(dims)

        rads = [t for t in terms if t.kind == CollapseTermKind.RADIATIVE]
        self.assertEqual(len(rads), 4)

        for t in rads:
            self.assertEqual(t.meta["type"], "radiative")
            initial = t.meta["initial"]
            final = t.meta["final"]
            # op should be |final><initial|
            expected = Qobj(op_matrix(final, initial), dims=[dims, dims]).to(
                "csr"
            )
            self.assertLess((t.op - expected).norm(), 1e-12)

    def test_phonon_dephasing_are_projectors(self):
        dims = [4]
        with patch(
            "bec.quantum_dot.me.base_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            terms = self.cb.build_catalog(dims)

        ph = [t for t in terms if t.kind == CollapseTermKind.PHONON]
        self.assertEqual({t.meta["level"] for t in ph}, {"X1", "X2", "XX"})

        for t in ph:
            P = t.op
            self.assertTrue(P.isherm)
            self.assertLess((P * P - P).norm(), 1e-12)


if __name__ == "__main__":
    unittest.main()
