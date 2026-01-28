import unittest

from smef.core.units import Q, as_quantity, magnitude

from bec.light.classical.carrier_profiles import (
    constant,
    linear_chirp,
    tanh_chirp,
)


class TestCarrierProfiles(unittest.TestCase):
    def test_constant(self) -> None:
        fn = constant(Q(12.0, "rad/s"))
        y0 = fn(Q(0.0, "s"))
        y1 = fn(Q(5.0, "s"))

        self.assertAlmostEqual(float(magnitude(y0, "rad/s")), 12.0, places=12)
        self.assertAlmostEqual(float(magnitude(y1, "rad/s")), 12.0, places=12)

    def test_linear_chirp_zero_at_t0(self) -> None:
        rate = Q(3.0, "rad/s^2")
        t0 = Q(10.0, "ps")
        fn = linear_chirp(rate=rate, t0=t0)

        y = fn(t0)
        self.assertAlmostEqual(float(magnitude(y, "rad/s")), 0.0, places=12)

    def test_linear_chirp_sign_and_scale(self) -> None:
        rate = Q(2.0, "rad/s^2")
        t0 = Q(0.0, "s")
        fn = linear_chirp(rate=rate, t0=t0)

        y_pos = fn(Q(1.5, "s"))  # 2.0 * 1.5 = 3.0 rad/s
        y_neg = fn(Q(-2.0, "s"))  # 2.0 * (-2.0) = -4.0 rad/s

        self.assertAlmostEqual(float(magnitude(y_pos, "rad/s")), 3.0, places=12)
        self.assertAlmostEqual(
            float(magnitude(y_neg, "rad/s")), -4.0, places=12
        )

    def test_tanh_chirp_center_and_limits(self) -> None:
        t0 = Q(0.0, "s")
        dm = Q(10.0, "rad/s")
        tau = Q(2.0, "s")
        fn = tanh_chirp(t0=t0, delta_max=dm, tau=tau)

        y0 = fn(Q(0.0, "s"))
        self.assertAlmostEqual(float(magnitude(y0, "rad/s")), 0.0, places=12)

        y_far_pos = fn(Q(100.0, "s"))
        y_far_neg = fn(Q(-100.0, "s"))

        self.assertAlmostEqual(
            float(magnitude(y_far_pos, "rad/s")), 10.0, places=10
        )
        self.assertAlmostEqual(
            float(magnitude(y_far_neg, "rad/s")), -10.0, places=10
        )

        # boundedness sanity check
        for t in [Q(-3.0, "s"), Q(-1.0, "s"), Q(0.5, "s"), Q(4.0, "s")]:
            y = float(magnitude(fn(t), "rad/s"))
            self.assertLessEqual(y, 10.0 + 1e-12)
            self.assertGreaterEqual(y, -10.0 - 1e-12)

    def test_tanh_chirp_tau_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            _ = tanh_chirp(
                t0=Q(0.0, "s"), delta_max=Q(1.0, "rad/s"), tau=Q(0.0, "s")
            )

        with self.assertRaises(ValueError):
            _ = tanh_chirp(
                t0=Q(0.0, "s"), delta_max=Q(1.0, "rad/s"), tau=Q(-1.0, "s")
            )


if __name__ == "__main__":
    unittest.main()
