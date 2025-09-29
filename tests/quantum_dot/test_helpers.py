import unittest
from types import SimpleNamespace

from bec.quantum_dot.helpers import infer_index_sets_from_registry
from bec.params.transitions import Transition, TransitionType


class FakeModesContainer:
    """Holds .modes and implements by_transition_and_source."""

    def __init__(self, modes, N_cut=None):
        self.modes = modes[:]  # list of objects with .transition, .source
        if N_cut is not None:
            self.N_cut = N_cut

    def by_transition_and_source(self, transition, source):
        for i, m in enumerate(self.modes):
            if m.transition == transition and m.source == source:
                return i, m
        raise ValueError("mode not found")


def make_qd(modes, qd_N_cut=None, cavity_N_cut=None, modes_N_cut=None):
    qd = SimpleNamespace()
    qd.modes = FakeModesContainer(modes, N_cut=modes_N_cut)
    if qd_N_cut is not None:
        qd.N_cut = qd_N_cut
    if cavity_N_cut is not None:
        qd.cavity_params = SimpleNamespace(N_cut=cavity_N_cut)
    return qd


def mode(transition):
    return SimpleNamespace(
        transition=transition, source=TransitionType.INTERNAL
    )


class InferIndexSetsTests(unittest.TestCase):
    def test_four_mode_layout_no_offset_and_explicit_dim(self):
        modes = [
            mode(Transition.X1_XX),
            mode(Transition.X2_XX),
            mode(Transition.G_X1),
            mode(Transition.G_X2),
        ]
        qd = make_qd(modes)

        d = 4  # factor_dim explicitly provided
        early, late, plus, minus, dims, offset = infer_index_sets_from_registry(
            qd, rho_has_qd=False, factor_dim=d
        )

        # Expected indices with offset=0:
        # plus_index  = 0 + 2*i
        # minus_index = 0 + 2*i + 1
        e1p, e1m = 0, 1
        e2p, e2m = 2, 3
        l1p, l1m = 4, 5  # for i=2
        l2p, l2m = 6, 7  # for i=3

        self.assertEqual(early, [e1p, e1m, e2p, e2m])
        self.assertEqual(late, [l1p, l1m, l2p, l2m])
        self.assertEqual(plus, [e1p, e2p, l1p, l2p])
        self.assertEqual(minus, [e1m, e2m, l1m, l2m])

        self.assertEqual(offset, 0)
        self.assertEqual(len(dims), len(early) + len(late))
        self.assertTrue(all(x == d for x in dims))

    def test_two_mode_layout_with_qd_offset_and_inferred_dim_from_qd(self):
        # Mode indices: i=0:X_XX, i=1:G_X
        modes = [mode(Transition.X_XX), mode(Transition.G_X)]
        qd = make_qd(modes, qd_N_cut=7)  # factor_dim should infer to 7

        early, late, plus, minus, dims, offset = infer_index_sets_from_registry(
            qd, rho_has_qd=True, factor_dim=None
        )

        # offset=1 -> plus_index = 1 + 2*i
        ep, em = 1, 2  # i=0
        lp, lm = 3, 4  # i=1

        self.assertEqual(early, [ep, em])
        self.assertEqual(late, [lp, lm])
        self.assertEqual(plus, [ep, lp])
        self.assertEqual(minus, [em, lm])

        self.assertEqual(offset, 1)
        self.assertEqual(len(dims), 4)
        self.assertTrue(all(x == 7 for x in dims))

    def test_infer_dim_from_cavity_params_when_qd_N_cut_missing(self):
        modes = [mode(Transition.X_XX), mode(Transition.G_X)]
        qd = make_qd(modes, cavity_N_cut=5)  # infer from cavity_params.N_cut

        _, _, _, _, dims, _ = infer_index_sets_from_registry(
            qd, rho_has_qd=False, factor_dim=None
        )
        self.assertEqual(len(dims), 4)
        self.assertTrue(all(x == 5 for x in dims))

    def test_infer_dim_from_modes_container_when_others_missing(self):
        modes = [mode(Transition.X_XX), mode(Transition.G_X)]
        qd = make_qd(modes, modes_N_cut=6)  # infer from qd.modes.N_cut

        _, _, _, _, dims, _ = infer_index_sets_from_registry(
            qd, rho_has_qd=False, factor_dim=None
        )
        self.assertEqual(len(dims), 4)
        self.assertTrue(all(x == 6 for x in dims))

    def test_raises_when_layout_cannot_be_inferred(self):
        # Include only X1_XX and G_X (neither full 4-mode nor valid 2-mode)
        modes = [mode(Transition.X1_XX), mode(Transition.G_X)]
        qd = make_qd(modes)

        with self.assertRaises(ValueError):
            _ = infer_index_sets_from_registry(
                qd, rho_has_qd=False, factor_dim=3
            )


if __name__ == "__main__":
    unittest.main()
