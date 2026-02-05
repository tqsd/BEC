import unittest

from bec.quantum_dot.enums import (
    QDState,
    RateKey,
    Transition,
    TransitionKind,
    TransitionPair,
    directed_of,
    reverse_transition,
    transition_pair_of,
)


class TestQuantumDotEnums(unittest.TestCase):
    def test_enum_values_are_strings(self) -> None:
        # Ensures str/Enum mixin behaves as expected for JSON/reporting keys
        self.assertIsInstance(QDState.G.value, str)
        self.assertIsInstance(Transition.G_X1.value, str)
        self.assertIsInstance(TransitionPair.G_X1.value, str)
        self.assertIsInstance(TransitionKind.DIPOLE_1PH.value, str)
        self.assertIsInstance(RateKey.RAD_X1_G.value, str)

    def test_qdstate_members(self) -> None:
        self.assertEqual(
            set(QDState), {QDState.G, QDState.X1, QDState.X2, QDState.XX}
        )

    def test_transition_members_include_expected(self) -> None:
        # Spot checks for regressions / accidental renaming
        self.assertIn(Transition.G_X1, list(Transition))
        self.assertIn(Transition.X1_G, list(Transition))
        self.assertIn(Transition.G_XX, list(Transition))
        self.assertIn(Transition.XX_G, list(Transition))

    def test_transitionpair_members(self) -> None:
        self.assertIn(TransitionPair.G_X1, list(TransitionPair))
        self.assertIn(TransitionPair.G_XX, list(TransitionPair))

    def test_transitionkind_members(self) -> None:
        self.assertEqual(TransitionKind.DIPOLE_1PH.value, "dipole_1ph")
        self.assertEqual(TransitionKind.EFFECTIVE_2PH.value, "effective_2ph")

    def test_ratekey_members(self) -> None:
        # Spot check
        self.assertEqual(RateKey.RAD_XX_X1.value, "RAD_XX_X1")
        self.assertEqual(RateKey.PH_DEPH_X1.value, "PH_DEPH_X1")

    def test_transition_pair_mapping(self) -> None:
        # Pair mapping should map both directions to the same pair
        self.assertEqual(
            transition_pair_of(Transition.G_X1), TransitionPair.G_X1
        )
        self.assertEqual(
            transition_pair_of(Transition.X1_G), TransitionPair.G_X1
        )

        self.assertEqual(
            transition_pair_of(Transition.X2_XX), TransitionPair.X2_XX
        )
        self.assertEqual(
            transition_pair_of(Transition.XX_X2), TransitionPair.X2_XX
        )

    def test_reverse_transition(self) -> None:
        self.assertEqual(reverse_transition(Transition.G_X1), Transition.X1_G)
        self.assertEqual(reverse_transition(Transition.X1_G), Transition.G_X1)

        self.assertEqual(reverse_transition(Transition.G_XX), Transition.XX_G)
        self.assertEqual(reverse_transition(Transition.XX_G), Transition.G_XX)

    def test_directed_of_pair(self) -> None:
        fwd, bwd = directed_of(TransitionPair.G_X1)
        self.assertEqual(fwd, Transition.G_X1)
        self.assertEqual(bwd, Transition.X1_G)

        fwd, bwd = directed_of(TransitionPair.G_XX)
        self.assertEqual(fwd, Transition.G_XX)
        self.assertEqual(bwd, Transition.XX_G)


if __name__ == "__main__":
    unittest.main()
