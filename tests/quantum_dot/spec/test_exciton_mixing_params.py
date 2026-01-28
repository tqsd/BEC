import unittest

from smef.core.units import Q

from bec.quantum_dot.spec.exciton_mixing_params import ExcitonMixingParams


class TestExcitonMixingParams(unittest.TestCase):
    def test_default(self) -> None:
        p = ExcitonMixingParams()
        self.assertAlmostEqual(
            float(p.delta_prime.to("eV").magnitude), 0.0, places=12
        )

    def test_from_values_float(self) -> None:
        p = ExcitonMixingParams.from_values(delta_prime_eV=0.001)
        self.assertAlmostEqual(
            float(p.delta_prime.to("eV").magnitude), 0.001, places=12
        )

    def test_from_values_quantity(self) -> None:
        p = ExcitonMixingParams.from_values(delta_prime_eV=Q(2.0, "meV"))
        self.assertAlmostEqual(
            float(p.delta_prime.to("eV").magnitude), 0.002, places=12
        )

    def test_reject_nan(self) -> None:
        with self.assertRaises(ValueError):
            ExcitonMixingParams.from_values(delta_prime_eV=float("nan"))


if __name__ == "__main__":
    unittest.main()
