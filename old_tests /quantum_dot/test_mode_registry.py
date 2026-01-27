import unittest
from types import SimpleNamespace
from enum import Enum, auto

from bec.quantum_dot.mode_registry import ModeRegistry


# Local lightweight enums to avoid importing the real ones
class Transition(Enum):
    G_X1 = auto()
    G_X2 = auto()
    X1_XX = auto()
    X2_XX = auto()


class TransitionType(Enum):
    INTERNAL = auto()
    EXTERNAL = auto()


def LM(transition=None, source=None, label=None):
    """Convenience: make a lightweight LightMode-like object."""
    return SimpleNamespace(transition=transition, source=source, label=label)


class TestModeRegistry(unittest.TestCase):
    def setUp(self):
        # two intrinsic modes
        self.intrinsic = [
            LM(
                transition=Transition.G_X1,
                source=TransitionType.INTERNAL,
                label="i0",
            ),
            LM(
                transition=Transition.G_X2,
                source=TransitionType.INTERNAL,
                label="i1",
            ),
        ]
        self.theta, self.phi = 0.3, 1.1
        self.reg = ModeRegistry(self.intrinsic, (self.theta, self.phi))

    def test_rotation_params_are_stored(self):
        self.assertEqual(self.reg.THETA, self.theta)
        self.assertEqual(self.reg.PHI, self.phi)

    def test_modes_returns_intrinsic_then_external_and_is_copy(self):
        # initially only intrinsic
        m0 = self.reg.modes
        self.assertEqual([m.label for m in m0], ["i0", "i1"])

        # add two external modes
        e0 = LM(
            transition=Transition.X1_XX,
            source=TransitionType.EXTERNAL,
            label="e0",
        )
        e1 = LM(
            transition=Transition.X2_XX,
            source=TransitionType.EXTERNAL,
            label="e1",
        )
        self.reg.register_external(e0)
        self.reg.register_external(e1)

        m1 = self.reg.modes
        self.assertEqual([m.label for m in m1], ["i0", "i1", "e0", "e1"])

        # mutating returned list must not affect internal state
        m1.append(LM(label="should_not_appear"))
        self.assertEqual(
            [m.label for m in self.reg.modes], ["i0", "i1", "e0", "e1"]
        )

    def test_register_external_and_reset(self):
        self.assertEqual(len(self.reg.modes), 2)
        ex = LM(
            transition=Transition.X1_XX,
            source=TransitionType.EXTERNAL,
            label="ex",
        )
        self.reg.register_external(ex)
        self.assertEqual([m.label for m in self.reg.modes], ["i0", "i1", "ex"])

        # reset clears only externals
        self.reg.reset()
        self.assertEqual([m.label for m in self.reg.modes], ["i0", "i1"])

    def test_by_transition_and_source_returns_index_and_mode(self):
        # Add an external too
        ex = LM(
            transition=Transition.X1_XX,
            source=TransitionType.EXTERNAL,
            label="ex",
        )
        self.reg.register_external(ex)

        # intrinsic match
        idx, mode = self.reg.by_transition_and_source(
            Transition.G_X2, TransitionType.INTERNAL
        )
        # i1 at index 1 in concatenated list
        self.assertEqual(idx, 1)
        self.assertEqual(mode.label, "i1")

        # external match
        idx2, mode2 = self.reg.by_transition_and_source(
            Transition.X1_XX, TransitionType.EXTERNAL
        )
        self.assertEqual(idx2, 2)  # ex comes after 2 intrinsics
        self.assertEqual(mode2.label, "ex")

    def test_by_transition_prefers_intrinsic_when_both_exist(self):
        # Register an external that duplicates an intrinsic transition/source pair? (rare)
        # More realistic: duplicate transition but different source; however, test first-match rule:
        dup_intrinsic = LM(
            transition=Transition.G_X1,
            source=TransitionType.INTERNAL,
            label="dup_i",
        )
        # Put it into *external* list to ensure it would appear after intrinsics anyway
        self.reg.register_external(dup_intrinsic)

        idx, mode = self.reg.by_transition_and_source(
            Transition.G_X1, TransitionType.INTERNAL
        )
        # Should return the intrinsic one first (index 0), not the later duplicate
        self.assertEqual(idx, 0)
        self.assertEqual(mode.label, "i0")

    def test_by_transition_and_source_raises_when_not_found(self):
        with self.assertRaises(ValueError):
            self.reg.by_transition_and_source(
                Transition.X2_XX, TransitionType.INTERNAL
            )


if __name__ == "__main__":
    unittest.main()
