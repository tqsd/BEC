import unittest
import numpy as np
from qutip import Qobj
from unittest.mock import patch

from scipy.constants import e as _e, hbar as _hbar

from bec.quantum_dot.me.hamiltonian_builder import HamiltonianBuilder
from bec.quantum_dot.me.types import HamiltonianTermKind


LEVELS = ["G", "X1", "X2", "XX"]
IDX = {name: i for i, name in enumerate(LEVELS)}


def basis_ket(i: int, N: int = 4) -> np.ndarray:
    v = np.zeros((N, 1), dtype=complex)
    v[i, 0] = 1.0
    return v


def op_matrix(bra: str, ket: str) -> np.ndarray:
    i = IDX[bra]
    j = IDX[ket]
    return basis_ket(i) @ basis_ket(j).conj().T  # |bra><ket|


class FakeKronPad:
    def pad(self, local_op, subsystem_label, idx):
        # dot-only embedding: do nothing
        return local_op


class FakeEnergyLevels:
    def __init__(self, fss: float = 0.0, delta_prime: float = 0.0):
        self.fss = fss
        self.delta_prime = delta_prime


def make_fake_ctx():
    ctx = {}
    for bra in LEVELS:
        for ket in LEVELS:
            mat = op_matrix(bra, ket)
            # context functions are called like ctx[key]([])
            ctx[f"s_{bra}_{ket}"] = lambda _ignored, m=mat: m
    return ctx


class TestHamiltonianBuilder(unittest.TestCase):
    def setUp(self):
        self.ctx = make_fake_ctx()
        self.kron = FakeKronPad()
        # Put in some nonzero numbers so the fss test is meaningful
        self.EL = FakeEnergyLevels(fss=2e-6, delta_prime=1e-6)
        self.builder = HamiltonianBuilder(
            context=self.ctx,
            kron=self.kron,
            energy_levels=self.EL,
            pm_map=lambda i: "+",  # unused in current builder
        )

    def test_catalog_counts_and_kinds(self):
        dims = [4]
        time_unit_s = 1.0

        # Patch interpreter used inside the module under test.
        # IMPORTANT: patch the symbol where it is USED, i.e. in bec.quantum_dot.hamiltonian_builder
        with patch(
            "bec.quantum_dot.hamiltonian_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            terms = self.builder.build_catalog(dims, time_unit_s)

        self.assertEqual(len(terms), 17)
        self.assertEqual(
            sum(t.kind == HamiltonianTermKind.STATIC for t in terms), 1
        )
        self.assertEqual(
            sum(t.kind == HamiltonianTermKind.DETUNING for t in terms), 4
        )
        self.assertEqual(
            sum(t.kind == HamiltonianTermKind.DRIVE for t in terms), 12
        )

    def test_all_ops_have_correct_dims(self):
        dims = [4]
        with patch(
            "bec.quantum_dot.hamiltonian_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            terms = self.builder.build_catalog(dims, time_unit_s=1.0)

        for t in terms:
            self.assertIsInstance(t.op, Qobj)
            self.assertEqual(t.op.dims, [dims, dims])

    def test_projectors_are_hermitian_and_idempotent(self):
        dims = [4]
        with patch(
            "bec.quantum_dot.hamiltonian_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            terms = self.builder.build_catalog(dims, time_unit_s=1.0)

        projs = [t for t in terms if t.kind == HamiltonianTermKind.DETUNING]
        self.assertEqual({t.meta["level"] for t in projs}, set(LEVELS))

        for t in projs:
            P = t.op
            self.assertTrue(P.isherm)
            self.assertLess((P * P - P).norm(), 1e-12)

    def test_coherences_have_dagger_pair(self):
        dims = [4]
        with patch(
            "bec.quantum_dot.hamiltonian_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            terms = self.builder.build_catalog(dims, time_unit_s=1.0)

        coh = {
            (t.meta["bra"], t.meta["ket"]): t.op
            for t in terms
            if t.kind == HamiltonianTermKind.DRIVE
        }

        for bra in LEVELS:
            for ket in LEVELS:
                if bra == ket:
                    continue
                self.assertIn((bra, ket), coh)
                self.assertIn((ket, bra), coh)
                self.assertLess(
                    (coh[(bra, ket)].dag() - coh[(ket, bra)]).norm(), 1e-12
                )

    def test_fss_matches_formula(self):
        dims = [4]
        time_unit_s = 1.0

        with patch(
            "bec.quantum_dot.hamiltonian_builder.interpreter",
            side_effect=lambda expr, ctx, dims: expr,
        ):
            terms = self.builder.build_catalog(dims, time_unit_s=time_unit_s)

        fss_term = next(
            t
            for t in terms
            if t.kind == HamiltonianTermKind.STATIC and t.label == "fss"
        )
        proj = {
            t.meta["level"]: t.op
            for t in terms
            if t.kind == HamiltonianTermKind.DETUNING
        }
        coh = {
            (t.meta["bra"], t.meta["ket"]): t.op
            for t in terms
            if t.kind == HamiltonianTermKind.DRIVE
        }

        Delta = self.EL.fss * _e / _hbar * time_unit_s
        Delta_p = self.EL.delta_prime * _e / _hbar * time_unit_s

        expected = (Delta / 2) * (proj["X1"] - proj["X2"]) + (Delta_p / 2) * (
            coh[("X1", "X2")] + coh[("X2", "X1")]
        )

        self.assertLess((fss_term.op - expected).norm(), 1e-12)


if __name__ == "__main__":
    unittest.main()
