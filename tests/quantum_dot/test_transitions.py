import unittest

from bec.quantum_dot.enums import (
    QDState,
    Transition,
    TransitionKind,
    TransitionPair,
)
from bec.quantum_dot.transitions import (
    DEFAULT_TRANSITION_REGISTRY,
    TransitionRegistry,
)


class TestTransitionRegistry(unittest.TestCase):
    def test_default_registry_endpoints(self) -> None:
        reg = DEFAULT_TRANSITION_REGISTRY
        self.assertEqual(
            reg.endpoints(Transition.G_X1), (QDState.G, QDState.X1)
        )
        self.assertEqual(
            reg.endpoints(Transition.XX_X2), (QDState.XX, QDState.X2)
        )

    def test_reverse_is_involution(self) -> None:
        reg = DEFAULT_TRANSITION_REGISTRY
        for tr in Transition:
            self.assertEqual(reg.reverse(reg.reverse(tr)), tr)

    def test_as_pair_and_directed(self) -> None:
        reg = DEFAULT_TRANSITION_REGISTRY
        self.assertEqual(reg.as_pair(Transition.G_X1), TransitionPair.G_X1)
        fwd, bwd = reg.directed(TransitionPair.G_X1)
        self.assertEqual(fwd, Transition.G_X1)
        self.assertEqual(bwd, Transition.X1_G)

    def test_pair_endpoints(self) -> None:
        reg = DEFAULT_TRANSITION_REGISTRY
        self.assertEqual(
            reg.pair_endpoints(TransitionPair.G_X2), (QDState.G, QDState.X2)
        )

    def test_specs_exist(self) -> None:
        reg = DEFAULT_TRANSITION_REGISTRY
        s = reg.spec(TransitionPair.G_X1)
        self.assertEqual(s.kind, TransitionKind.DIPOLE_1PH)
        self.assertEqual(s.order, 1)
        self.assertTrue(s.decay_allowed)

        s2 = reg.spec(TransitionPair.G_XX)
        self.assertEqual(s2.kind, TransitionKind.EFFECTIVE_2PH)
        self.assertEqual(s2.order, 2)
        self.assertFalse(s2.decay_allowed)

        # Spec can be queried by Transition too
        s3 = reg.spec(Transition.G_XX)
        self.assertEqual(s3.kind, TransitionKind.EFFECTIVE_2PH)

    def test_from_states(self) -> None:
        reg = DEFAULT_TRANSITION_REGISTRY
        tr = reg.from_states(QDState.X2, QDState.G)
        self.assertEqual(tr, Transition.X2_G)

        missing = reg.from_states(QDState.X1, QDState.X2)
        self.assertIsNone(missing)

    def test_validation_catches_inconsistent_pair(self) -> None:
        # Build a broken registry: swap an endpoint so forward/backward don't match
        # type: ignore[attr-defined]
        endpoints = dict(DEFAULT_TRANSITION_REGISTRY._endpoints)
        endpoints[Transition.X1_G] = (QDState.X2, QDState.G)  # wrong

        with self.assertRaises(ValueError):
            TransitionRegistry.build(
                endpoints=endpoints,
                # type: ignore[attr-defined]
                pair_to_directed=dict(DEFAULT_TRANSITION_REGISTRY._pair_to_dir),
                # type: ignore[attr-defined]
                specs=dict(DEFAULT_TRANSITION_REGISTRY._specs),
            )


if __name__ == "__main__":
    unittest.main()
