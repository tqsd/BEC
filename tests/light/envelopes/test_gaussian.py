import unittest
import math

from smef.core.units import Q

from bec.light.envelopes.gaussian import GaussianEnvelopeU


class TestGaussianEnvelopeU(unittest.TestCase):
    def test_sigma_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            GaussianEnvelopeU(t0=Q(0.0, "s"), sigma=Q(0.0, "s"))
        with self.assertRaises(ValueError):
            GaussianEnvelopeU(t0=Q(0.0, "s"), sigma=Q(-1.0, "ps"))

    def test_peak_is_one_at_t0(self) -> None:
        env = GaussianEnvelopeU(t0=Q(5.0, "ps"), sigma=Q(2.0, "ps"))
        y = env(Q(5.0, "ps"))
        self.assertAlmostEqual(y, 1.0, places=12)

    def test_symmetry_about_t0(self) -> None:
        env = GaussianEnvelopeU(t0=Q(10.0, "ps"), sigma=Q(2.0, "ps"))
        y1 = env(Q(11.0, "ps"))
        y2 = env(Q(9.0, "ps"))
        self.assertAlmostEqual(y1, y2, places=12)

    def test_strict_unitful_time_input(self) -> None:
        env = GaussianEnvelopeU(t0=Q(0.0, "s"), sigma=Q(1.0, "s"))
        with self.assertRaises(TypeError):
            env(0.0)  # type: ignore[arg-type]

    def test_area_seconds(self) -> None:
        env = GaussianEnvelopeU(t0=Q(0.0, "s"), sigma=Q(2.0, "s"))
        area = env.area_seconds()
        self.assertAlmostEqual(area, 2.0 * math.sqrt(2.0 * math.pi), places=12)

    def test_from_fwhm(self) -> None:
        t0 = Q(0.0, "ps")
        fwhm = Q(10.0, "ps")
        env = GaussianEnvelopeU.from_fwhm(t0=t0, fwhm=fwhm)

        # sigma = fwhm / (2*sqrt(2*ln(2)))
        expected_sigma_ps = 10.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        self.assertAlmostEqual(
            float(env.sigma.to("ps").magnitude), expected_sigma_ps, places=12
        )

        # peak still at t0
        self.assertAlmostEqual(env(Q(0.0, "ps")), 1.0, places=12)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        env = GaussianEnvelopeU(t0=Q(5.0, "ps"), sigma=Q(2.0, "ps"))
        d = env.to_dict()
        env2 = GaussianEnvelopeU.from_dict(d)

        self.assertAlmostEqual(
            float(env2.t0.to("ps").magnitude), 5.0, places=12
        )
        self.assertAlmostEqual(
            float(env2.sigma.to("ps").magnitude), 2.0, places=12
        )
        self.assertAlmostEqual(env2(Q(5.0, "ps")), 1.0, places=12)

    def test_internal_cached_seconds(self) -> None:
        env = GaussianEnvelopeU(t0=Q(5.0, "ps"), sigma=Q(2.0, "ps"))
        # sanity check that caches exist and match expected seconds
        self.assertAlmostEqual(getattr(env, "_t0_s"), 5.0e-12, places=30)
        self.assertAlmostEqual(getattr(env, "_sig_s"), 2.0e-12, places=30)


if __name__ == "__main__":
    unittest.main()
