import unittest
from types import SimpleNamespace

import numpy as np

from bec.operators.qd_operators import QDState
from bec.params.energy_levels import EnergyLevels
from bec.quantum_dot.models.decay_model import DecayModel


class TestDecayModel(unittest.TestCase):
    """
    Unit tests for DecayModel.

    Notes
    -----
    - EnergyLevels is a dataclass with a @property `energies_eV`, so you cannot
      "inject" a dict into it. All tests must construct EnergyLevels via its
      constructor parameters (biexciton, exciton, fss, ...).
    - EnergyLevels enforces a 2-photon-regime guard by default; in unit tests
      we disable it unless we explicitly want to test that guard.
    """

    # A physically reasonable default set (binding energy ~ 3 meV, small FSS)
    EXCITON = 1.300  # eV (mean exciton energy)
    BINDING = 3e-3  # eV (3 meV binding energy)
    FSS = 5e-6  # eV (~5 µeV)

    def _make_energy_levels(
        self,
        *,
        biexciton: float | None = None,
        exciton: float | None = None,
        fss: float | None = None,
        delta_prime: float = 0.0,
        enforce_2g_guard: bool = False,
        min_binding_energy_meV: float = 1.0,
        pulse_sigma_t_s: float | None = None,
    ) -> EnergyLevels:
        """
        Construct an EnergyLevels instance with sensible defaults.

        Parameters are the *actual* EnergyLevels constructor arguments.
        """
        exciton = float(self.EXCITON if exciton is None else exciton)
        fss = float(self.FSS if fss is None else fss)

        # Default biexciton chosen to yield the requested binding energy:
        # binding = (X1 + X2) - XX = 2*exciton - XX  (since X1+X2 = 2*exciton)
        if biexciton is None:
            biexciton = float(2.0 * exciton - self.BINDING)

        return EnergyLevels(
            biexciton=float(biexciton),
            exciton=exciton,
            fss=fss,
            delta_prime=float(delta_prime),
            enforce_2g_guard=bool(enforce_2g_guard),
            min_binding_energy_meV=float(min_binding_energy_meV),
            pulse_sigma_t_s=pulse_sigma_t_s,
        )

    def _make_dipole(self, mu_Cm: float):
        return SimpleNamespace(dipole_moment_Cm=float(mu_Cm))

    def _make_cavity(self, n=3.5, Veff_um3=1.0, Q=10_000):
        return SimpleNamespace(n=float(n), Veff_um3=float(Veff_um3), Q=float(Q))

    def test_init_requires_dipole_params(self):
        el = self._make_energy_levels()
        with self.assertRaises(ValueError):
            DecayModel(el, cavity_params=None, dipole_params=None)

    def test_omega_non_positive_energy_gap_returns_zero(self):
        el = self._make_energy_levels()
        model = DecayModel(
            el, cavity_params=None, dipole_params=self._make_dipole(1e-29)
        )

        self.assertEqual(model._omega(1.0, 1.0), 0.0)
        self.assertEqual(model._omega(0.9, 1.0), 0.0)

    def test_gamma0_zero_for_invalid_inputs(self):
        el = self._make_energy_levels()
        model = DecayModel(
            el, cavity_params=None, dipole_params=self._make_dipole(1e-29)
        )

        self.assertEqual(model._gamma0(omega=0.0, mu_Cm=1e-29), 0.0)
        self.assertEqual(model._gamma0(omega=-1.0, mu_Cm=1e-29), 0.0)
        self.assertEqual(model._gamma0(omega=1.0, mu_Cm=0.0), 0.0)
        self.assertEqual(model._gamma0(omega=1.0, mu_Cm=-1.0), 0.0)

    def test_purcell_zero_without_cavity_or_invalid_omega(self):
        el = self._make_energy_levels()
        model = DecayModel(
            el, cavity_params=None, dipole_params=self._make_dipole(1e-29)
        )

        self.assertEqual(model._purcell(omega=0.0), 0.0)
        self.assertEqual(model._purcell(omega=-1.0), 0.0)

    def test_compute_returns_expected_keys_and_nonnegative_rates(self):
        el = self._make_energy_levels()
        dipole = self._make_dipole(mu_Cm=1e-29)
        model = DecayModel(el, cavity_params=None, dipole_params=dipole)

        rates = model.compute()

        self.assertEqual(
            set(rates.keys()), {"L_XX_X1", "L_XX_X2", "L_X1_G", "L_X2_G"}
        )
        for k, v in rates.items():
            self.assertIsInstance(v, float)
            self.assertTrue(np.isfinite(v), msg=f"{k} not finite")
            self.assertGreaterEqual(v, 0.0, msg=f"{k} negative")

    def test_compute_gives_zero_rate_if_transition_not_downhill(self):
        # Force XX below X1/X2 by setting biexciton < exciton.
        # This is unphysical for the cascade but exercises the omega guard:
        # if Ei <= Ef => omega=0 => gamma=0.
        el = self._make_energy_levels(
            biexciton=1.0, exciton=2.0, fss=0.0, enforce_2g_guard=False
        )
        dipole = self._make_dipole(mu_Cm=1e-29)
        model = DecayModel(el, cavity_params=None, dipole_params=dipole)

        rates = model.compute()
        self.assertEqual(rates["L_XX_X1"], 0.0)
        self.assertEqual(rates["L_XX_X2"], 0.0)

    def test_purcell_increases_rates_when_cavity_present(self):
        el = self._make_energy_levels()
        dipole = self._make_dipole(mu_Cm=1e-29)

        no_cav = DecayModel(
            el, cavity_params=None, dipole_params=dipole
        ).compute()
        cav = DecayModel(
            el, cavity_params=self._make_cavity(Q=50_000), dipole_params=dipole
        ).compute()

        for key in no_cav.keys():
            self.assertGreaterEqual(
                cav[key], no_cav[key], msg=f"{key} not increased by cavity"
            )

    def test_compute_scales_with_dipole_moment_squared(self):
        el = self._make_energy_levels()

        mu1 = 1e-29
        mu2 = 2e-29  # 2x dipole => 4x gamma

        r1 = DecayModel(
            el, cavity_params=None, dipole_params=self._make_dipole(mu1)
        ).compute()
        r2 = DecayModel(
            el, cavity_params=None, dipole_params=self._make_dipole(mu2)
        ).compute()

        ratio_expected = (mu2 / mu1) ** 2
        for key in r1.keys():
            if r1[key] == 0.0:
                continue
            # Use a looser delta than 1e-12 to avoid platform-dependent float noise
            self.assertAlmostEqual(
                r2[key] / r1[key],
                ratio_expected,
                delta=1e-10,
                msg=f"{key} scaling wrong",
            )

    def test_energylevels_guard_can_be_enabled(self):
        # When enforce_2g_guard=True and binding energy is too small, EnergyLevels should raise.
        # Construct a near-zero binding energy case: XX ≈ 2*exciton.
        with self.assertRaises(Exception):
            self._make_energy_levels(
                biexciton=2.0 * self.EXCITON,  # binding ~ 0
                exciton=self.EXCITON,
                fss=0.0,
                enforce_2g_guard=True,
                min_binding_energy_meV=1.0,
                pulse_sigma_t_s=None,
            )


if __name__ == "__main__":
    unittest.main()
