import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from qutip import Qobj

from bec.quantum_dot.hamiltonian_builder import HamiltonianBuilder


# ---- helpers / fakes ---------------------------------------------------------


class FakeModes:
    def __init__(self, modes):
        self.modes = modes[:]  # list of objects with .label and .transitions


class FakeKronPad:
    """
    Mimic the API used by HamiltonianBuilder. Exposes
      - pad(qd_op, fock, idx) -> ("PAD", qd_op, fock, idx)
      - by_label(label) -> index
      - _modes (with .modes)
    """

    def __init__(self, modes):
        self._modes = FakeModes(modes)

    def pad(self, qd_op, fock, idx):
        return ("PAD", qd_op, fock, idx)

    def by_label(self, label: str) -> int:
        for i, m in enumerate(self._modes.modes):
            if getattr(m, "label", None) == label:
                return i
        raise ValueError(f"no mode with label {label!r}")


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
        # also used by classical helpers
        "s_G_XX": lambda _: np.eye(4, dtype=complex),  # arbitrary Hermitian
        "s_XX_G": lambda _: np.eye(4, dtype=complex),  # arbitrary Hermitian
        "s_XX_XX": lambda _: np.diag([0, 0, 0, 1]).astype(complex),
        # lmi needs these symbols by *name* (they're not called here directly)
        "s_G_X1": lambda _: np.array(
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=complex,
        ),
        "s_G_X2": lambda _: np.array(
            [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=complex,
        ),
        "s_X1_G": lambda _: np.array(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=complex,
        ),
        "s_X2_G": lambda _: np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            dtype=complex,
        ),
        "s_X1_XX": lambda _: np.zeros((4, 4), dtype=complex),
        "s_X2_XX": lambda _: np.zeros((4, 4), dtype=complex),
        "s_XX_X1": lambda _: np.zeros((4, 4), dtype=complex),
        "s_XX_X2": lambda _: np.zeros((4, 4), dtype=complex),
    }


class EnergyLevelsStub:
    def __init__(self, fss: float, delta_prime: float):
        self.fss = fss
        self.delta_prime = delta_prime


def pm_map(idx: int) -> str | None:
    # Same rule as in your system: 0,2 -> '+', 1,3 -> '-'
    return "+" if idx in (0, 2) else "-" if idx in (1, 3) else None


# ---- tests -------------------------------------------------------------------


class HamiltonianBuilderTests(unittest.TestCase):
    def setUp(self):
        # Two modes: one 'single' (for lmi), one 'tpe' (for tpe)
        self.modes = [
            SimpleNamespace(
                label="m_single", transitions=[0, 3]
            ),  # exercises '+' and '-'
            SimpleNamespace(label="m_tpe", transitions=[4]),
        ]
        self.kron = FakeKronPad(self.modes)
        self.ctx = make_ctx_for_fss()
        self.EL = EnergyLevelsStub(fss=0.25, delta_prime=0.10)
        self.builder = HamiltonianBuilder(self.ctx, self.kron, self.EL, pm_map)

        # Composite dims (QD + 1 mode for example); only product matters to Qobj
        self.dims = [4, 2, 2]
        self.N = int(np.prod(self.dims))

    @patch("bec.quantum_dot.hamiltonian_builder.interpreter")
    def test_fss_builds_expected_local_term_and_uses_kron_pad(
        self, interpreter_mock
    ):
        # Make interpreter return identity of correct size
        interpreter_mock.side_effect = lambda expr, ctx, dims: np.eye(
            self.N, dtype=complex
        )

        Hq = self.builder.fss(self.dims)
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

        # Validate Hloc numerically: (Δ/2)(X1-X2) + (Δ'/2)(X1X2+X2X1)
        Delta, Delta_p = self.EL.fss, self.EL.delta_prime
        X1X1 = self.ctx["s_X1_X1"]([])
        X2X2 = self.ctx["s_X2_X2"]([])
        X1X2 = self.ctx["s_X1_X2"]([])
        X2X1 = self.ctx["s_X2_X1"]([])
        Hloc_expected = (Delta / 2) * (X1X1 - X2X2) + (Delta_p / 2) * (
            X1X2 + X2X1
        )

        # expr is ("PAD", Hloc, "i", -1)
        Hloc_got = expr[1]
        np.testing.assert_allclose(
            Hloc_got, Hloc_expected, atol=1e-12, rtol=1e-12
        )

    @patch("bec.quantum_dot.hamiltonian_builder.interpreter")
    def test_lmi_assembles_expected_nested_expression(self, interpreter_mock):
        interpreter_mock.side_effect = lambda expr, ctx, dims: np.eye(
            self.N, dtype=complex
        )

        Hq = self.builder.lmi("m_single", self.dims)
        self.assertIsInstance(Hq, Qobj)
        self.assertEqual(Hq.shape, (self.N, self.N))

        # Check the expression structure passed to interpreter
        self.assertEqual(interpreter_mock.call_count, 1)
        expr, ctx, dims = interpreter_mock.call_args.args
        self.assertEqual(expr[0], "add")
        self.assertIs(ctx, self.ctx)
        self.assertEqual(dims, self.dims)

        # Build our expected H_ints for transitions [0, 3]
        # Map lookup arrays (by *name*; kron.pad returns ("PAD", name, fock, idx))
        ab = ["s_G_X1", "s_G_X2", "s_X1_XX", "s_X2_XX"]
        em = ["s_X1_G", "s_X2_G", "s_XX_X1", "s_XX_X2"]
        idx = self.kron.by_label("m_single")

        expected_terms = []
        for i in [0, 3]:
            pm = pm_map(i)
            expected_terms.append(
                (
                    "s_mult",
                    1,
                    (
                        "add",
                        ("PAD", em[i], f"a{pm}_dag", idx),
                        ("PAD", ab[i], f"a{pm}", idx),
                    ),
                )
            )

        # expr = ("add", *H_ints)
        got_terms = list(expr[1:])
        self.assertEqual(got_terms, expected_terms)

    @patch("bec.quantum_dot.hamiltonian_builder.interpreter")
    def test_tpe_uses_aa_and_aa_dag(self, interpreter_mock):
        interpreter_mock.side_effect = lambda expr, ctx, dims: np.eye(
            self.N, dtype=complex
        )
        Hq = self.builder.tpe("m_tpe", self.dims)
        self.assertIsInstance(Hq, Qobj)
        self.assertEqual(Hq.shape, (self.N, self.N))

        self.assertEqual(interpreter_mock.call_count, 1)
        expr, ctx, dims = interpreter_mock.call_args.args
        self.assertEqual(expr[0], "add")
        idx = self.kron.by_label("m_tpe")
        self.assertEqual(expr[1], ("PAD", "s_G_XX", "aa", idx))
        self.assertEqual(expr[2], ("PAD", "s_XX_G", "aa_dag", idx))

    @patch("bec.quantum_dot.hamiltonian_builder.interpreter")
    def test_classical_2g_helpers(self, interpreter_mock):
        interpreter_mock.side_effect = lambda expr, ctx, dims: np.eye(
            self.N, dtype=complex
        )

        Hflip = self.builder.classical_2g_flip(self.dims)
        Hdet = self.builder.classical_2g_detuning(self.dims)
        self.assertIsInstance(Hflip, Qobj)
        self.assertIsInstance(Hdet, Qobj)
        self.assertEqual(Hflip.shape, (self.N, self.N))
        self.assertEqual(Hdet.shape, (self.N, self.N))

        # Two separate calls
        self.assertEqual(interpreter_mock.call_count, 2)

        # First call: flip -> pad(H_GXX + H_XXG, "i", -1)
        expr1, ctx1, dims1 = interpreter_mock.call_args_list[0].args
        self.assertEqual(expr1[0], "PAD")
        self.assertEqual(expr1[2:], ("i", -1))
        # second element is Hloc = s_G_XX + s_XX_G
        Hloc1 = expr1[1]
        Hloc_expected = self.ctx["s_G_XX"]([]) + self.ctx["s_XX_G"]([])
        np.testing.assert_allclose(Hloc1, Hloc_expected, atol=1e-12, rtol=1e-12)
        self.assertIs(ctx1, self.ctx)
        self.assertEqual(dims1, self.dims)

        # Second call: detuning -> pad(s_XX_XX, "i", -1)
        expr2, ctx2, dims2 = interpreter_mock.call_args_list[1].args

        # tuple shape parts
        self.assertEqual(expr2[0], "PAD")
        self.assertEqual(expr2[2:], ("i", -1))
        self.assertIs(ctx2, self.ctx)
        self.assertEqual(dims2, self.dims)

        # the matrix payload
        np.testing.assert_allclose(
            expr2[1],
            self.ctx["s_XX_XX"]([]),
            atol=1e-12,
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
