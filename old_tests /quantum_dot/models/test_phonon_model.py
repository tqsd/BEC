import unittest
from unittest.mock import patch

import numpy as np

from bec.params.energy_levels import EnergyLevels
from bec.params.phonon_params import PhononParams, PhononModelType
from bec.quantum_dot.models.phonon_model import PhononModel, PhononOutputs


class TestPhononModel(unittest.TestCase):
    def _make_energy_levels(self) -> EnergyLevels:
        # Keep the 2-gamma guard off for unit tests.
        return EnergyLevels(
            biexciton=2.0 * 1.300 - 3e-3,  # binding ~ 3 meV
            exciton=1.300,
            fss=5e-6,
            delta_prime=0.0,
            enforce_2g_guard=False,
        )

    def test_compute_returns_defaults_when_params_none(self):
        el = self._make_energy_levels()
        pm = PhononModel(el, PP=None)

        out = pm.compute()
        self.assertIsInstance(out, PhononOutputs)
        self.assertEqual(out.B_polaron, 1.0)
        self.assertEqual(out.rates_1_s, {})

    def test_polaron_B_returns_1_when_params_none(self):
        el = self._make_energy_levels()
        pm = PhononModel(el, PP=None)
        self.assertEqual(pm.polaron_B(), 1.0)

    def test_polaron_B_returns_1_when_alpha_is_zero(self):
        el = self._make_energy_levels()
        pp = PhononParams(
            model=PhononModelType.POLARON,
            temperature_K=4.0,
            enable_polaron_renorm=True,
            alpha_s2=0.0,
            omega_c_rad_s=1.0e12,
        )
        pm = PhononModel(el, PP=pp)
        self.assertEqual(pm.polaron_B(), 1.0)

    def test_polaron_B_returns_finite_and_in_0_1_for_reasonable_params(self):
        el = self._make_energy_levels()
        pp = PhononParams(
            model=PhononModelType.POLARON,
            temperature_K=4.0,
            enable_polaron_renorm=True,
            alpha_s2=0.03e-24,  # positive coupling
            omega_c_rad_s=1.0e12,
        )
        pm = PhononModel(el, PP=pp)

        B = pm.polaron_B()
        self.assertTrue(np.isfinite(B))
        self.assertGreaterEqual(B, 0.0)
        self.assertLessEqual(B, 1.0)

        # With positive coupling, polaron dressing should usually reduce B below 1
        # (not guaranteed for extreme edge cases, but typical).
        self.assertLessEqual(B, 1.0)

    def test_polaron_B_temperature_leq_zero_branch_is_safe(self):
        el = self._make_energy_levels()
        pp = PhononParams(
            model=PhononModelType.POLARON,
            temperature_K=0.0,  # triggers eta = inf branch
            enable_polaron_renorm=True,
            alpha_s2=0.03e-24,
            omega_c_rad_s=1.0e12,
        )
        pm = PhononModel(el, PP=pp)

        B = pm.polaron_B()
        self.assertTrue(np.isfinite(B))
        self.assertGreaterEqual(B, 0.0)
        self.assertLessEqual(B, 1.0)

    def test_polaron_B_cache_is_used(self):
        el = self._make_energy_levels()
        pp = PhononParams(
            model=PhononModelType.POLARON,
            temperature_K=4.0,
            enable_polaron_renorm=True,
            alpha_s2=0.03e-24,
            omega_c_rad_s=1.0e12,
        )
        pm = PhononModel(el, PP=pp)

        self.assertEqual(len(pm._cache), 0)
        B1 = pm.polaron_B()
        self.assertGreaterEqual(len(pm._cache), 1)
        cache_size = len(pm._cache)

        B2 = pm.polaron_B()
        self.assertEqual(len(pm._cache), cache_size)
        self.assertEqual(B1, B2)

    def test_compute_includes_only_positive_phenomenological_rates(self):
        el = self._make_energy_levels()
        pp = PhononParams(
            model=PhononModelType.PHENOMENOLOGICAL,
            temperature_K=4.0,
            gamma_phi_Xp_1_s=1.0,
            gamma_phi_Xm_1_s=0.0,
            gamma_phi_XX_1_s=2.0,
            enable_polaron_renorm=True,  # ignored in PHENOMENOLOGICAL mode
        )
        pm = PhononModel(el, PP=pp)

        out = pm.compute()
        self.assertEqual(
            out.B_polaron, 1.0
        )  # compute() only applies B in POLARON+enabled
        self.assertEqual(set(out.rates_1_s.keys()), {"Lphi_Xp", "Lphi_XX"})
        self.assertEqual(out.rates_1_s["Lphi_Xp"], 1.0)
        self.assertEqual(out.rates_1_s["Lphi_XX"], 2.0)

    def test_compute_calls_polaron_B_only_when_polaron_and_enabled(self):
        el = self._make_energy_levels()
        pp = PhononParams(
            model=PhononModelType.POLARON,
            temperature_K=4.0,
            enable_polaron_renorm=True,
            alpha_s2=0.03e-24,
            omega_c_rad_s=1e12,
        )
        pm = PhononModel(el, PP=pp)

        with patch.object(pm, "polaron_B", return_value=0.123) as mocked:
            out = pm.compute()
            mocked.assert_called_once()
            self.assertAlmostEqual(out.B_polaron, 0.123)
            # no phenomenological rates set
            self.assertEqual(out.rates_1_s, {})

    def test_compute_does_not_call_polaron_B_when_disabled(self):
        el = self._make_energy_levels()
        pp = PhononParams(
            model=PhononModelType.POLARON,
            temperature_K=4.0,
            enable_polaron_renorm=False,  # explicitly disabled
            alpha_s2=0.03e-24,
            omega_c_rad_s=1e12,
        )
        pm = PhononModel(el, PP=pp)

        with patch.object(pm, "polaron_B", return_value=0.123) as mocked:
            out = pm.compute()
            mocked.assert_not_called()
            self.assertEqual(out.B_polaron, 1.0)

    def test_compute_does_not_call_polaron_B_if_model_not_polaron(self):
        el = self._make_energy_levels()
        pp = PhononParams(
            model=PhononModelType.PHENOMENOLOGICAL,
            temperature_K=4.0,
            enable_polaron_renorm=True,
            alpha_s2=0.03e-24,
            omega_c_rad_s=1e12,
        )
        pm = PhononModel(el, PP=pp)

        with patch.object(pm, "polaron_B", return_value=0.123) as mocked:
            out = pm.compute()
            mocked.assert_not_called()
            self.assertEqual(out.B_polaron, 1.0)

    def test_phononparams_validation_negative_values_raise(self):
        with self.assertRaises(ValueError):
            PhononParams(temperature_K=-1.0)
        with self.assertRaises(ValueError):
            PhononParams(gamma_phi_Xp_1_s=-1.0)
        with self.assertRaises(ValueError):
            PhononParams(alpha_s2=-1.0)

    def test_phononparams_validation_omega_c_zero_when_renorm_enabled_raises(
        self,
    ):
        with self.assertRaises(ValueError):
            PhononParams(
                model=PhononModelType.POLARON,
                enable_polaron_renorm=True,
                omega_c_rad_s=0.0,
            )


if __name__ == "__main__":
    unittest.main()
