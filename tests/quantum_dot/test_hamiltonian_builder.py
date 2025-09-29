import unittest
from unittest.mock import patch
import numpy as np
from qutip import Qobj

from bec.quantum_dot.hamiltonian_builder import HamiltonianBuilder


class FakeKronPad:
    """
    Minimal KronPad stub exposing pad(...) and by_label(...).
    Only pad(...) is used by the tests here.
    """

    def __init__(self):
        self._modes = type("M", (), {"modes": []})()  # not used

    def pad(self, qd_op, fock, idx):
        # Match the tuple structure the builder expects to hand to interpreter
        return ("PAD", qd_op, fock, idx)

    def by_label(self, label: str) -> int:
        raise NotImplementedError("by_label not used in these tests")


def make_ctx_for_fss():
    """
    Build a context dict where each value is a lambda that ignores its arg and
    returns a fixed 4x4 array for the named operator.
    """

    def proj(i):
        M = np.zeros((4, 4), dtype=complex)
        M[i, i] = 1.0
        return M

    # Choose X1=|1>, X2=|2>
    X1X1 = proj(1)
    X2X2 = proj(2)
    X1X2 = np.zeros((4, 4), dtype=complex)
    X1X2[1, 2] = 1.0
    X2X1 = np.zeros((4, 4), dtype=complex)
    X2X1[2, 1] = 1.0

    return {
        "s_X1_X1": lambda _: X1X1,
        "s_X2_X2": lambda _: X2X2,
        "s_X1_X2": lambda _: X1X2,
        "s_X2_X1": lambda _: X2X1,
        # classical helpers
        "s_G_XX": lambda _: np.eye(4, dtype=complex),
        "s_XX_G": lambda _: np.eye(4, dtype=complex),
        "s_XX_XX": lambda _: np.diag([0, 0, 0, 1]).astype(complex),
    }


class EnergyLevelsStub:
    def __init__(self, fss: float, delta_prime: float):
        # values in eV, as expected by HamiltonianBuilder
        self.fss = fss
        self.delta_prime = delta_prime


class HamiltonianBuilderTests(unittest.TestCase):
    def setUp(self):
        self.kron = FakeKronPad()
        self.ctx = make_ctx_for_fss()
        # Use simple eV values; test will apply e/ħ * time_unit_s scaling
        self.EL = EnergyLevelsStub(fss=0.25, delta_prime=0.10)
        self.builder = HamiltonianBuilder(
            self.ctx, self.kron, self.EL, pm_map=None
        )

        # Composite dims (QD + one small mode for example); only product matters
        self.dims = [4, 2]
        self.N = int(np.prod(self.dims))

    @patch("bec.quantum_dot.hamiltonian_builder.interpreter")
    def test_fss_builds_expected_local_term_and_uses_kron_pad(
        self, interpreter_mock
    ):
        # Make interpreter return an identity of the correct size
        interpreter_mock.side_effect = lambda expr, ctx, dims: np.eye(
            self.N, dtype=complex
        )

        time_unit_s = 1.5e-12
        Hq = self.builder.fss(self.dims, time_unit_s)
        self.assertIsInstance(Hq, Qobj)
        self.assertEqual(Hq.shape, (self.N, self.N))
        self.assertEqual(Hq.dims, [self.dims, self.dims])

        # interpreter called once with ("PAD", Hloc, "i", -1)
        self.assertEqual(interpreter_mock.call_count, 1)
        expr, ctx, dims = interpreter_mock.call_args.args
        self.assertEqual(expr[0], "PAD")
        self.assertEqual(expr[2:], ("i", -1))
        self.assertIs(ctx, self.ctx)
        self.assertEqual(dims, self.dims)

        from scipy.constants import e as _e, hbar as _hbar

        Delta = self.EL.fss * _e / _hbar * time_unit_s
        Delta_p = self.EL.delta_prime * _e / _hbar * time_unit_s

        X1X1 = self.ctx["s_X1_X1"]([])
        X2X2 = self.ctx["s_X2_X2"]([])
        X1X2 = self.ctx["s_X1_X2"]([])
        X2X1 = self.ctx["s_X2_X1"]([])

        Hloc_expected = (Delta / 2) * (X1X1 - X2X2) + (Delta_p / 2) * (
            X1X2 + X2X1
        )
        Hloc_got = expr[1]  # payload inside ("PAD", Hloc, "i", -1)

        np.testing.assert_allclose(
            Hloc_got, Hloc_expected, atol=1e-12, rtol=1e-12
        )

    @patch("bec.quantum_dot.hamiltonian_builder.interpreter")
    def test_classical_2g_flip_and_detuning_build_expected_payload(
        self, interpreter_mock
    ):
        interpreter_mock.side_effect = lambda expr, ctx, dims: np.eye(
            self.N, dtype=complex
        )

        # flip
        Hflip = self.builder.classical_2g_flip(self.dims)
        self.assertIsInstance(Hflip, Qobj)
        self.assertEqual(Hflip.shape, (self.N, self.N))

        # detuning
        Hdet = self.builder.classical_2g_detuning(self.dims)
        self.assertIsInstance(Hdet, Qobj)
        self.assertEqual(Hdet.shape, (self.N, self.N))

        # Two interpreter calls: first flip, then detuning
        self.assertEqual(interpreter_mock.call_count, 2)

        expr1, ctx1, dims1 = interpreter_mock.call_args_list[0].args
        self.assertEqual(expr1[0], "PAD")
        self.assertEqual(expr1[2:], ("i", -1))
        self.assertIs(ctx1, self.ctx)
        self.assertEqual(dims1, self.dims)
        Hloc1 = expr1[1]
        Hloc1_expected = 0.5 * (self.ctx["s_G_XX"]([]) + self.ctx["s_XX_G"]([]))
        np.testing.assert_allclose(
            Hloc1, Hloc1_expected, atol=1e-12, rtol=1e-12
        )

        expr2, ctx2, dims2 = interpreter_mock.call_args_list[1].args
        self.assertEqual(expr2[0], "PAD")
        self.assertEqual(expr2[2:], ("i", -1))
        self.assertIs(ctx2, self.ctx)
        self.assertEqual(dims2, self.dims)
        Hloc2 = expr2[1]
        Hloc2_expected = 0.5 * self.ctx["s_XX_XX"]([])
        np.testing.assert_allclose(
            Hloc2, Hloc2_expected, atol=1e-12, rtol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
