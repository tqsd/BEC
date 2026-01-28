import unittest

from smef.core.units import Q

from bec.light.envelopes.symbolic import SymbolicEnvelopeU


class TestSymbolicEnvelopeU(unittest.TestCase):
    def test_eval_seconds_default(self) -> None:
        env = SymbolicEnvelopeU(expr="np.exp(-t*t)", params={})
        y = env(Q(2.0, "s"))
        # exp(-4)
        self.assertAlmostEqual(y, 0.01831563888873418, places=12)

    def test_eval_with_t_unit_ps(self) -> None:
        # Here t is in ps, so at t=2 ps => exp(-4)
        env = SymbolicEnvelopeU(expr="np.exp(-t*t)", params={}, t_unit="ps")
        y = env(Q(2.0, "ps"))
        self.assertAlmostEqual(y, 0.01831563888873418, places=12)

        # If we pass 2 ps but interpret as seconds, this would be exp(-4e-24) ~ 1
        env_s = SymbolicEnvelopeU(expr="np.exp(-t*t)", params={}, t_unit="s")
        y_s = env_s(Q(2.0, "ps"))
        self.assertAlmostEqual(y_s, 1.0, places=12)

    def test_params_are_available(self) -> None:
        env = SymbolicEnvelopeU(
            expr="a * np.exp(-t/b)", params={"a": 2.0, "b": 4.0}
        )
        y = env(Q(4.0, "s"))
        # 2 * exp(-1)
        self.assertAlmostEqual(y, 2.0 / 2.718281828459045, places=12)

    def test_strict_unitful_time_input(self) -> None:
        env = SymbolicEnvelopeU(expr="t", params={})
        with self.assertRaises(TypeError):
            env(1.0)  # type: ignore[arg-type]

    def test_to_dict_from_dict_roundtrip(self) -> None:
        env = SymbolicEnvelopeU(
            expr="np.sin(t) + k", params={"k": 0.25}, t_unit="ps"
        )
        d = env.to_dict()
        env2 = SymbolicEnvelopeU.from_dict(d)

        self.assertEqual(env2.expr, env.expr)
        self.assertEqual(env2.t_unit, "ps")
        self.assertAlmostEqual(env2.params["k"], 0.25, places=12)

        y1 = env(Q(1.0, "ps"))
        y2 = env2(Q(1.0, "ps"))
        self.assertAlmostEqual(y1, y2, places=12)

    def test_restricted_eval_blocks_builtins(self) -> None:
        # Attempt to access builtins; should fail.
        env = SymbolicEnvelopeU(
            expr="__import__('os').system('echo nope')", params={}
        )
        with self.assertRaises(Exception):
            _ = env(Q(0.0, "s"))


if __name__ == "__main__":
    unittest.main()
