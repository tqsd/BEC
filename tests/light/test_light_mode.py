import unittest
import importlib

from bec.light import LightMode
from bec.params.transitions import TransitionRole


class TestLightMode(unittest.TestCase):
    def setUp(self):
        # Find the module where LightMode is defined (so we can monkeypatch the converters)
        self.light_mod = importlib.import_module(LightMode.__module__)

        # Save originals to restore later
        self.orig_e2w = getattr(self.light_mod, "energy_to_wavelength_nm")
        self.orig_w2e = getattr(self.light_mod, "wavelength_to_energy_eV")

        # Simple, exact, invertible fns for deterministic tests:
        # wavelength_nm = energy_eV * 1000
        # energy_eV = wavelength_nm / 1000
        setattr(
            self.light_mod,
            "energy_to_wavelength_nm",
            lambda e: float(e) * 1000.0,
        )
        setattr(
            self.light_mod,
            "wavelength_to_energy_eV",
            lambda w: float(w) / 1000.0,
        )

    def tearDown(self):
        # Restore original converter functions
        setattr(self.light_mod, "energy_to_wavelength_nm", self.orig_e2w)
        setattr(self.light_mod, "wavelength_to_energy_eV", self.orig_w2e)

    def test_init_with_energy_sets_wavelength_via_converter(self):
        m = LightMode(energy_ev=2.0)
        self.assertIsNotNone(m.wavelength_nm)
        self.assertAlmostEqual(m.wavelength_nm, 2000.0, places=12)
        self.assertAlmostEqual(m.energy_ev, 2.0, places=12)

    def test_init_with_wavelength_sets_energy_via_converter(self):
        m = LightMode(wavelength_nm=1500.0)
        self.assertIsNotNone(m.energy_ev)
        self.assertAlmostEqual(m.energy_ev, 1.5, places=12)
        self.assertAlmostEqual(m.wavelength_nm, 1500.0, places=12)

    def test_error_when_both_none(self):
        with self.assertRaises(ValueError):
            _ = LightMode()  # neither wavelength nor energy

    def test_error_when_both_provided(self):
        with self.assertRaises(ValueError):
            _ = LightMode(wavelength_nm=800.0, energy_ev=1.55)

    def test_equality_is_id_based_not_value_based(self):
        m1 = LightMode(wavelength_nm=1000.0)
        m2 = LightMode(wavelength_nm=1000.0)
        self.assertNotEqual(m1, m2)  # different internal IDs
        self.assertEqual(m1, m1)  # reflexive
        self.assertEqual(m2, m2)

    def test_hash_and_set_membership(self):
        m1 = LightMode(energy_ev=1.0)
        m2 = LightMode(energy_ev=1.0)
        s = {m1, m2}
        # different IDs -> distinct set entries
        self.assertEqual(len(s), 2)
        self.assertIn(m1, s)
        self.assertIn(m2, s)

    def test_defaults_role_and_tpe(self):
        m = LightMode(wavelength_nm=900.0)
        self.assertEqual(m.role, TransitionRole.SINGLE)
        self.assertEqual(m.tpe_eliminated, set())
        self.assertAlmostEqual(m.tpe_alpha_X1, 0.0, places=12)
        self.assertAlmostEqual(m.tpe_alpha_X2, 0.0, places=12)

    def test_hash_uses_id_not_only_wavelength(self):
        # Sanity check that hash can differ even if wavelength is same
        m1 = LightMode(wavelength_nm=1550.0)
        m2 = LightMode(wavelength_nm=1550.0)
        self.assertNotEqual(hash(m1), hash(m2))


if __name__ == "__main__":
    unittest.main()
