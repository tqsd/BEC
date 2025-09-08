import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from qutip import Qobj

from bec.quantum_dot.observables_builder import ObservablesBuilder
from bec.quantum_dot.kron_pad_utility import KronPad


class FakeModeProvider:
    def __init__(self, labels):
        self.modes = [SimpleNamespace(label=lb) for lb in labels]


def make_ctx():
    # Only used as a token passed through to interpreter; content isn't used
    return {
        "s_X1_X1": lambda _: np.diag([0, 1, 0, 0]).astype(complex),
        "s_X2_X2": lambda _: np.diag([0, 0, 1, 0]).astype(complex),
        "s_G_G": lambda _: np.diag([1, 0, 0, 0]).astype(complex),
        "s_XX_XX": lambda _: np.diag([0, 0, 0, 1]).astype(complex),
        "idq": lambda _: np.eye(4, dtype=complex),
    }


class ObservablesBuilderTests(unittest.TestCase):
    def setUp(self):
        # Two modes → dims has QD + 2*2 pol blocks = [4, 2, 2, 2, 2]
        self.provider = FakeModeProvider(["m0", "m1"])
        self.kron = KronPad(self.provider)
        self.ctx = make_ctx()
        self.builder = ObservablesBuilder(self.ctx, self.kron, self.provider)
        self.dims = [4, 2, 2, 2, 2]
        self.N = int(np.prod(self.dims))

    # ---------------- QD projectors ----------------

    @patch("bec.quantum_dot.observables_builder.interpreter")
    def test_qd_projectors_builds_four_qobjs_and_calls_interpreter_with_kron(
        self, interpreter_mock
    ):
        # Make interpreter return identity so Qobj construction succeeds
        interpreter_mock.side_effect = lambda expr, ctx, dims: np.eye(
            self.N, dtype=complex
        )

        out = self.builder.qd_projectors(self.dims)

        # Four projectors with correct shapes
        self.assertEqual(set(out.keys()), {"P_G", "P_X1", "P_X2", "P_XX"})
        for q in out.values():
            self.assertIsInstance(q, Qobj)
            self.assertEqual(q.shape, (self.N, self.N))
            self.assertEqual(q.dims, [self.dims, self.dims])

        # Called once per projector
        self.assertEqual(interpreter_mock.call_count, 4)

        # Each call should be ("kron", local_qd_proj, "if0","if1")
        # because there are 2 modes → two "if" factors
        for call in interpreter_mock.call_args_list:
            expr, ctx, dims = call.args
            self.assertEqual(expr[0], "kron")
            # check trailing "if{i}" list length equals number of modes
            op_tail = expr[2:]
            self.assertEqual(op_tail, ("if0", "if1"))
            self.assertIs(ctx, self.ctx)
            self.assertEqual(dims, self.dims)
            # expr[1] should be a 4x4 projector array
            self.assertEqual(np.array(expr[1]).shape, (4, 4))

    # --------------- Per-mode projectors -----------

    @patch("bec.quantum_dot.observables_builder.interpreter")
    def test_light_mode_projectors_requests_expected_expressions_and_returns_qobjs(
        self, interpreter_mock
    ):
        # Return identity for all expressions to keep arithmetic simple
        interpreter_mock.side_effect = lambda expr, ctx, dims: np.eye(
            self.N, dtype=complex
        )

        ops = self.builder.light_mode_projectors(self.dims)

        # Expect keys for each mode label
        expected_keys = set()
        for label in ["m0", "m1"]:
            expected_keys |= {
                f"N[{label}]",
                f"N+[{label}]",
                f"N-[{label}]",
                f"Pvac[{label}]",
                f"P10[{label}]",
                f"P01[{label}]",
                f"P11[{label}]",
                f"S0[{label}]",
                f"S1[{label}]",
            }
        self.assertEqual(set(ops.keys()), expected_keys)

        # All values are Qobj with correct dims
        for q in ops.values():
            self.assertIsInstance(q, Qobj)
            self.assertEqual(q.shape, (self.N, self.N))
            self.assertEqual(q.dims, [self.dims, self.dims])

        # Interpreter should be called 4 times per mode: n+, n-, vac, i
        self.assertEqual(
            interpreter_mock.call_count, 4 * len(self.provider.modes)
        )

        # Verify the exact expressions requested
        # Calls are in order: for mode 0 (n+, n-, vac, i), then mode 1 (n+, n-, vac, i)
        calls = interpreter_mock.call_args_list
        expected_exprs = [
            # mode 0
            ("kron", "idq", "n0+", "if1"),
            ("kron", "idq", "n0-", "if1"),
            ("kron", "idq", "vac0", "if1"),
            ("kron", "idq", "if0", "if1"),  # "i" → "if{idx}"
            # mode 1
            ("kron", "idq", "if0", "n1+"),
            ("kron", "idq", "if0", "n1-"),
            ("kron", "idq", "if0", "vac1"),
            ("kron", "idq", "if0", "if1"),
        ]

        # Extract simplified tuples from actual expr for comparison
        got_exprs = []
        for call in calls:
            expr, ctx, dims = call.args
            self.assertEqual(expr[0], "kron")
            self.assertIs(ctx, self.ctx)
            self.assertEqual(dims, self.dims)
            # expr is ("kron", qd_op, op0, op1, ...) where qd_op is "idq" here
            qd_part = expr[1] if isinstance(expr[1], str) else "idq"
            got_exprs.append(("kron", qd_part, *expr[2:]))

        self.assertEqual(got_exprs, expected_exprs)


if __name__ == "__main__":
    unittest.main()
