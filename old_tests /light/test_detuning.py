import unittest
import numpy as np
from scipy.constants import e as _e, hbar as _hbar

from bec.light.detuning import two_photon_detuning_profile
from bec.params.energy_levels import EnergyLevels
from bec.light.classical import ClassicalTwoPhotonDrive


class TestTwoPhotonDetuningProfile(unittest.TestCase):
    def setUp(self):
        # Simple constant envelope
        self.env = lambda t: 1.0

        # A small solver time grid
        self.tlist_solver = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)

        # Time scaling: seconds per solver time unit
        self.s = 2.0

        # Energy levels: choose easy numbers
        # XX = 1.5 eV, exciton and fss irrelevant for this function
        self.EL = EnergyLevels(
            biexciton=1.5, exciton=1.0, fss=0.0, enforce_2g_guard=False
        )

        # Convenience: two-photon resonance frequency in rad/s
        self.wXXG = float(self.EL.XX) * _e / _hbar

    def test_uses_laser_omega_when_available(self):
        # Target physical detuning
        Delta_phys = 3.2e9  # rad/s

        # Choose laser_omega so that 2*laser_omega - wXXG = Delta_phys
        laser_omega = 0.5 * (self.wXXG + Delta_phys)

        drive = ClassicalTwoPhotonDrive(
            envelope=self.env,
            omega0=1.0,
            detuning=0.0,
            laser_omega=laser_omega,
        ).with_cached_tlist(self.tlist_solver)

        t, Delta_solver = two_photon_detuning_profile(self.EL, drive, self.s)

        self.assertIsNotNone(t)
        np.testing.assert_allclose(t, self.tlist_solver, rtol=0, atol=0)

        # Expected solver detuning is Delta_phys * time_unit_s
        expected = np.full_like(
            self.tlist_solver, Delta_phys * self.s, dtype=float
        )
        np.testing.assert_allclose(Delta_solver, expected, rtol=0, atol=1e-12)

    def test_falls_back_to_float_detuning_when_no_laser_freq(self):
        # Physical detuning set directly
        Delta_phys = -7.5e9  # rad/s

        drive = ClassicalTwoPhotonDrive(
            envelope=self.env,
            omega0=1.0,
            detuning=Delta_phys,
            laser_omega=None,
        ).with_cached_tlist(self.tlist_solver)

        t, Delta_solver = two_photon_detuning_profile(self.EL, drive, self.s)

        expected = np.full_like(
            self.tlist_solver, Delta_phys * self.s, dtype=float
        )
        np.testing.assert_allclose(Delta_solver, expected, rtol=0, atol=1e-12)

    def test_callable_detuning_is_ignored_defaults_to_zero(self):
        # Provide a callable detuning; per function it should be treated as 0.0
        def det_fn(t):
            return 123.0  # should be ignored

        drive = ClassicalTwoPhotonDrive(
            envelope=self.env,
            omega0=1.0,
            detuning=det_fn,
            laser_omega=None,
        ).with_cached_tlist(self.tlist_solver)

        t, Delta_solver = two_photon_detuning_profile(self.EL, drive, self.s)

        expected = np.zeros_like(self.tlist_solver, dtype=float)  # 0.0 * s
        np.testing.assert_allclose(Delta_solver, expected, rtol=0, atol=1e-12)

    def test_returns_empty_arrays_when_cached_tlist_empty(self):
        # Default ClassicalTwoPhotonDrive has an empty cached t-list
        Delta_phys = 1.0e9
        drive = ClassicalTwoPhotonDrive(
            envelope=self.env,
            detuning=Delta_phys,
            laser_omega=None,
        )
        # Here _cached_tlist is an empty array, not None
        t, Delta_solver = two_photon_detuning_profile(self.EL, drive, self.s)

        # Expect empty arrays, matching the behavior of np.full_like on empty
        self.assertIsInstance(t, np.ndarray)
        self.assertEqual(t.size, 0)
        self.assertIsInstance(Delta_solver, np.ndarray)
        self.assertEqual(Delta_solver.size, 0)


if __name__ == "__main__":
    unittest.main()
