import unittest

from smef.core.units import Q

from bec.light.envelopes.gaussian import GaussianEnvelopeU
from bec.light.envelopes.registry import envelope_from_json, envelope_to_json
from bec.light.envelopes.symbolic import SymbolicEnvelopeU
from bec.light.envelopes.tabulated import TabulatedEnvelopeU


class TestEnvelopeRegistry(unittest.TestCase):
    def test_envelope_to_json_requires_type(self) -> None:
        class BadEnv:
            def to_dict(self):
                return {"foo": "bar"}

        with self.assertRaises(ValueError):
            envelope_to_json(BadEnv())  # type: ignore[arg-type]

    def test_envelope_from_json_requires_type_field(self) -> None:
        with self.assertRaises(TypeError):
            envelope_from_json({})  # missing "type"

        with self.assertRaises(TypeError):
            envelope_from_json({"type": 123})  # not a str

    def test_envelope_from_json_unknown_type(self) -> None:
        with self.assertRaises(ValueError):
            envelope_from_json({"type": "does_not_exist"})

    def test_gaussian_roundtrip(self) -> None:
        env = GaussianEnvelopeU(t0=Q(5.0, "ps"), sigma=Q(2.0, "ps"))
        data = envelope_to_json(env)
        env2 = envelope_from_json(data)

        self.assertIsInstance(env2, GaussianEnvelopeU)
        self.assertAlmostEqual(env2(Q(5.0, "ps")), 1.0, places=12)

    def test_symbolic_roundtrip(self) -> None:
        env = SymbolicEnvelopeU(
            expr="np.exp(-t*t) + k", params={"k": 0.25}, t_unit="ps"
        )
        data = envelope_to_json(env)
        env2 = envelope_from_json(data)

        self.assertIsInstance(env2, SymbolicEnvelopeU)
        y1 = env(Q(2.0, "ps"))
        y2 = env2(Q(2.0, "ps"))
        self.assertAlmostEqual(y1, y2, places=12)

    def test_tabulated_roundtrip(self) -> None:
        env = TabulatedEnvelopeU.from_samples(
            [Q(0.0, "ps"), Q(10.0, "ps"), Q(20.0, "ps")],
            [0.0, 1.0, 0.0],
            t_unit="ps",
        )
        data = envelope_to_json(env)
        env2 = envelope_from_json(data)

        self.assertIsInstance(env2, TabulatedEnvelopeU)
        self.assertAlmostEqual(env2(Q(10.0, "ps")), 1.0, places=12)
        self.assertAlmostEqual(env2(Q(5.0, "ps")), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
