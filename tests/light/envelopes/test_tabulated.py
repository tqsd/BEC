import unittest

from smef.core.units import Q

from bec.light.envelopes.tabulated import TabulatedEnvelopeU


class TestTabulatedEnvelopeU(unittest.TestCase):
    def test_requires_unitful_time(self) -> None:
        env = TabulatedEnvelopeU.from_samples(
            [Q(0.0, "s"), Q(1.0, "s")], [0.0, 1.0]
        )
        with self.assertRaises(TypeError):
            env(0.5)  # type: ignore[arg-type]

    def test_linear_interpolation(self) -> None:
        env = TabulatedEnvelopeU.from_samples(
            [Q(0.0, "s"), Q(1.0, "s"), Q(2.0, "s")],
            [0.0, 2.0, 0.0],
        )
        self.assertAlmostEqual(env(Q(0.0, "s")), 0.0, places=12)
        self.assertAlmostEqual(env(Q(0.5, "s")), 1.0, places=12)
        self.assertAlmostEqual(env(Q(1.0, "s")), 2.0, places=12)
        self.assertAlmostEqual(env(Q(1.5, "s")), 1.0, places=12)
        self.assertAlmostEqual(env(Q(2.0, "s")), 0.0, places=12)

    def test_left_right_clamping(self) -> None:
        env = TabulatedEnvelopeU.from_samples(
            [Q(1.0, "s"), Q(2.0, "s")], [3.0, 5.0]
        )
        self.assertAlmostEqual(env(Q(0.0, "s")), 3.0, places=12)
        self.assertAlmostEqual(env(Q(3.0, "s")), 5.0, places=12)

    def test_strictly_increasing_time(self) -> None:
        with self.assertRaises(ValueError):
            TabulatedEnvelopeU.from_samples(
                [Q(0.0, "s"), Q(0.0, "s")], [0.0, 1.0]
            )

    def test_to_dict_from_dict_roundtrip(self) -> None:
        env = TabulatedEnvelopeU.from_samples(
            [Q(0.0, "ps"), Q(10.0, "ps"), Q(20.0, "ps")],
            [0.0, 1.0, 0.0],
            t_unit="ps",
        )
        d = env.to_dict()
        env2 = TabulatedEnvelopeU.from_dict(d)

        self.assertEqual(env2.t_unit, "ps")
        self.assertAlmostEqual(env2(Q(10.0, "ps")), 1.0, places=12)
        self.assertAlmostEqual(env2(Q(5.0, "ps")), 0.5, places=12)

    def test_nan_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TabulatedEnvelopeU(t_s=(0.0, float("nan")), y=(0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
