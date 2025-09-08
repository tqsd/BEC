import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qutip import Qobj

from bec.quantum_dot.collapse_builder import CollapseBuilder
from bec.quantum_dot.kron_pad_utility import (
    KronPad,
)  # only for typing; faked below
from bec.params.transitions import Transition, TransitionType


class FakeModeProvider:
    def __init__(self, modes):
        self.modes = modes[:]  # list of objects having .source and .transition

    def by_transition_and_source(self, transition, source):
        for i, m in enumerate(self.modes):
            if m.transition == transition and m.source == source:
                return i, m
        raise ValueError(
            f"No mode with transition {
                         transition} and source {source}"
        )


class FakeKronPad:
    # Mimic the interface used by CollapseBuilder
    def pad(self, qd_op: str, fock: str, idx: int):
        # Return a simple structural tuple that we can assert against in tests
        return ("kron", qd_op, fock, idx)


class CollapseBuilderTests(unittest.TestCase):
    def setUp(self):
        # Gammas used by CollapseBuilder
        self.gammas = {
            "L_XX_X1": 1.6,  # sqrt = 1.264911064...
            "L_XX_X2": 2.5,  # sqrt = 1.581138830...
            "L_X1_G": 3.6,  # sqrt = 1.897366596...
            "L_X2_G": 4.0,  # sqrt = 2.0
        }
        # Context is unused by our mocked interpreter; keep minimal
        self.context = {}

        # Build mode list: four INTERNAL modes, one for each transition we need
        self.modes = [
            SimpleNamespace(
                source=TransitionType.INTERNAL, transition=Transition.X1_XX
            ),  # index 0
            SimpleNamespace(
                source=TransitionType.INTERNAL, transition=Transition.X2_XX
            ),  # index 1
            SimpleNamespace(
                source=TransitionType.INTERNAL, transition=Transition.G_X1
            ),  # index 2
            SimpleNamespace(
                source=TransitionType.INTERNAL, transition=Transition.G_X2
            ),  # index 3
        ]
        self.mode_provider = FakeModeProvider(self.modes)
        self.kron = FakeKronPad()

        self.builder = CollapseBuilder(
            gammas=self.gammas,
            context=self.context,
            kron=self.kron,
            mode_provider=self.mode_provider,
        )

        # Composite dimensions (QD + one mode for example); only the product matters to Qobj
        self.dims = [4, 2, 2]  # total Hilbert space dimension = 16

    @patch("photon_weave.extra.interpreter")
    def test_qutip_collapse_ops_calls_interpreter_with_expected_tuples(
        self, interpreter_mock
    ):
        # Make the interpreter return an array with the correct shape
        N = int(np.prod(self.dims))
        interpreter_mock.side_effect = lambda op, ctx, dims: np.eye(
            N, dtype=complex
        )

        ops = self.builder.qutip_collapse_ops(self.dims)

        # 1) We get four Qobj collapse operators
        self.assertEqual(len(ops), 4)
        for q in ops:
            self.assertIsInstance(q, Qobj)
            # QuTiP stores dims as [[...],[...]]
            self.assertEqual(q.dims, [self.dims, self.dims])
            self.assertEqual(q.shape, (N, N))

        # 2) Interpreter was called with the exact symbolic tuples, including sqrt gammas
        calls = interpreter_mock.call_args_list
        self.assertEqual(len(calls), 4)

        # Indices determined by FakeModeProvider order above
        i_xxx1 = 0
        i_xxx2 = 1
        i_x1g = 2
        i_x2g = 3

        expected_ops = [
            (
                "s_mult",
                np.sqrt(self.gammas["L_XX_X1"]),
                ("kron", "s_XX_X1", "a+_dag", i_xxx1),
            ),
            (
                "s_mult",
                np.sqrt(self.gammas["L_XX_X2"]),
                ("kron", "s_XX_X2", "a-_dag", i_xxx2),
            ),
            (
                "s_mult",
                np.sqrt(self.gammas["L_X1_G"]),
                ("kron", "s_X1_G", "a-_dag", i_x1g),
            ),
            (
                "s_mult",
                np.sqrt(self.gammas["L_X2_G"]),
                ("kron", "s_X2_G", "a+_dag", i_x2g),
            ),
        ]

        for call, expected in zip(calls, expected_ops):
            got_op, got_ctx, got_dims = call.args
            self.assertEqual(got_op, expected)
            self.assertIs(got_ctx, self.context)
            self.assertEqual(got_dims, self.dims)

    @patch("photon_weave.extra.interpreter")
    def test_missing_mode_raises_value_error(self, _interpreter_mock):
        # Remove one required mode so lookup fails
        self.mode_provider.modes = [
            m for m in self.modes if m.transition != Transition.X2_XX
        ]
        with self.assertRaises(ValueError):
            _ = self.builder.qutip_collapse_ops(self.dims)


if __name__ == "__main__":
    unittest.main()
