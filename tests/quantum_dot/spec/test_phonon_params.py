import unittest

from smef.core.units import Q

from bec.quantum_dot.spec.phonon_params import (
    PhononParams,
    PhononModelType,
    PhenomenologicalPhononParams,
    PolaronPhononParams,
)


class TestPhononParams(unittest.TestCase):
    def test_defaults(self) -> None:
        p = PhononParams()
        self.assertEqual(p.model, PhononModelType.POLARON)
        self.assertGreaterEqual(p.temperature_K, 0.0)

    def test_temperature_units(self) -> None:
        p = PhononParams(temperature=Q(10.0, "K"))
        self.assertAlmostEqual(p.temperature_K, 10.0, places=12)

    def test_negative_temperature_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PhononParams(temperature=Q(-1.0, "K"))

    def test_phenomenological_negative_rates_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PhenomenologicalPhononParams(gamma_phi_Xp=Q(-1.0, "1/s"))

    def test_polaron_omega_c_zero_rejected_when_renorm_enabled(self) -> None:
        with self.assertRaises(ValueError):
            PolaronPhononParams(
                enable_polaron_renorm=True, omega_c=Q(0.0, "rad/s")
            )

    def test_polaron_alpha_zero_allowed(self) -> None:
        p = PolaronPhononParams(alpha=Q(0.0, "s**2"))
        self.assertAlmostEqual(p.alpha_s2, 0.0, places=12)

    def test_as_floats_contains_expected_keys(self) -> None:
        p = PhononParams()
        d = p.as_floats()
        self.assertIn("temperature_K", d)
        self.assertIn("omega_c_rad_s", d)
        self.assertIn("alpha_s2", d)


if __name__ == "__main__":
    unittest.main()
