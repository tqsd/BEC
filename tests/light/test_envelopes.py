import unittest
import math
import numpy as np

from bec.light.envelopes import (
    Envelope,
    SerializableEnvelope,
    GaussianEnvelope,
    TabulatedEnvelope,
    SymbolicEnvelope,
    envelope_to_json,
    envelope_from_json,
    ENVELOPE_REGISTRY,
)


class TestGaussianEnvelope(unittest.TestCase):
    def test_value_at_center(self):
        g = GaussianEnvelope(t0=1.0, sigma=2.0, area=3.0)
        # At t0 the value is area / (sigma * sqrt(2*pi))
        expected = 3.0 / (2.0 * math.sqrt(2.0 * math.pi))
        self.assertAlmostEqual(g(1.0), expected, places=12)

    def test_from_fwhm_sigma_relation(self):
        fwhm = 5.0
        g = GaussianEnvelope.from_fwhm(t0=0.0, fwhm=fwhm, area=1.0)
        expected_sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        self.assertAlmostEqual(g.sigma, expected_sigma, places=12)

    def test_numerical_area_close(self):
        # Integrate over +/- 10 sigma; should capture almost all area
        area = 2.5
        g = GaussianEnvelope(t0=0.0, sigma=1.0, area=area)
        x = np.linspace(-10.0, 10.0, 20001)
        y = np.array([g(t) for t in x])
        trap = np.trapezoid(y, x)
        # Expect near area (remaining tails are negligible)
        self.assertAlmostEqual(trap, area, places=4)

    def test_roundtrip_json(self):
        g = GaussianEnvelope(t0=0.2, sigma=3.3, area=1.2)
        data = envelope_to_json(g)
        self.assertEqual(data["type"], "gaussian")
        g2 = envelope_from_json(data)
        self.assertEqual(g2, g)

    def test_protocol_checks(self):
        g = GaussianEnvelope(0.0, 1.0, 1.0)
        self.assertIsInstance(g, Envelope)
        self.assertIsInstance(g, SerializableEnvelope)


class TestTabulatedEnvelope(unittest.TestCase):
    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            TabulatedEnvelope(t=(0.0,), y=(1.0,))  # length < 2
        with self.assertRaises(ValueError):
            TabulatedEnvelope(t=(0.0, 1.0), y=(1.0,))  # length mismatch
        with self.assertRaises(ValueError):
            # not strictly increasing
            TabulatedEnvelope(t=(0.0, 0.0), y=(1.0, 2.0))
        with self.assertRaises(ValueError):
            TabulatedEnvelope(t=(0.0, 1.0), y=(float("nan"), 2.0))  # NaN in y
        with self.assertRaises(ValueError):
            TabulatedEnvelope(t=(0.0, float("nan")), y=(1.0, 2.0))  # NaN in t

    def test_interpolation_and_clamping(self):
        env = TabulatedEnvelope(t=(0.0, 1.0, 2.0), y=(0.0, 1.0, 0.0))
        # interior
        self.assertAlmostEqual(env(0.5), 0.5, places=12)
        self.assertAlmostEqual(env(1.5), 0.5, places=12)
        # clamping
        self.assertAlmostEqual(env(-10.0), 0.0, places=12)
        self.assertAlmostEqual(env(10.0), 0.0, places=12)

    def test_roundtrip_json(self):
        env = TabulatedEnvelope(t=(0.0, 1.0, 2.0), y=(0.0, 1.0, 0.0))
        data = env.to_dict()
        self.assertEqual(data["type"], "tabulated")
        env2 = TabulatedEnvelope.from_dict(data)
        self.assertEqual(env2, env)
        # Registry path
        env3 = envelope_from_json(data)
        self.assertEqual(env3, env)

    def test_protocol_checks(self):
        env = TabulatedEnvelope(t=(0.0, 1.0), y=(2.0, 3.0))
        self.assertIsInstance(env, Envelope)
        self.assertIsInstance(env, SerializableEnvelope)


class TestSymbolicEnvelope(unittest.TestCase):
    def test_eval_matches_manual(self):
        # A * exp(-((t - t0)^2)/(2*sigma^2)) * cos(omega*t + phi)
        params = {
            "A": 2.0,
            "t0": 0.5,
            "sigma": 0.2,
            "omega": 3.0,
            "phi": 0.1,
        }
        expr = (
            "A * math.exp(-((t - t0)**2)/(2*sigma**2)) * np.cos(omega*t + phi)"
        )
        env = SymbolicEnvelope(expr=expr, params=params)

        t = 0.7
        expected = (
            params["A"]
            * math.exp(-((t - params["t0"]) ** 2) / (2 * params["sigma"] ** 2))
            * math.cos(params["omega"] * t + params["phi"])
        )
        self.assertAlmostEqual(env(t), expected, places=12)

    def test_roundtrip_json(self):
        env = SymbolicEnvelope(expr="t + 1.0", params={})
        data = env.to_dict()
        self.assertEqual(data["type"], "symbolic")
        env2 = SymbolicEnvelope.from_dict(data)
        self.assertEqual(env2, env)
        # Registry path
        env3 = envelope_from_json(data)
        self.assertEqual(env3, env)

    def test_protocol_checks(self):
        env = SymbolicEnvelope(expr="t", params={})
        self.assertIsInstance(env, Envelope)
        self.assertIsInstance(env, SerializableEnvelope)


class TestRegistry(unittest.TestCase):
    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            envelope_from_json({"type": "nope"})

    def test_missing_type_raises(self):
        with self.assertRaises(ValueError):
            envelope_from_json({})

    def test_registry_contains_expected(self):
        # Basic sanity: keys present
        for k in ("gaussian", "tabulated", "symbolic"):
            self.assertIn(k, ENVELOPE_REGISTRY)
            cls = ENVELOPE_REGISTRY[k]
            # Ensure class advertises SerializableEnvelope
            self.assertTrue(issubclass(cls, SerializableEnvelope))


if __name__ == "__main__":
    unittest.main()
