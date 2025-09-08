import unittest
from types import SimpleNamespace
import numpy as np

from bec.quantum_dot.decay_model import DecayModel
from scipy.constants import epsilon_0, hbar, c, e  # for expected values


class TestDecayModel(unittest.TestCase):
    def setUp(self):
        # Representative level structure (eV)
        self.el = {
            "G": 0.0,
            "X1": 1.30,
            "X2": 1.29,
            "XX": 2.58,
        }
        # Time unit (seconds per simulation unit)
        self.dt = 1e-9  # ns

        # Minimal cavity and dipole containers
        self.no_cavity = None
        self.cavity = SimpleNamespace(
            Q=10_000.0, Veff_um3=1.0, n=3.5, lambda_nm=900.0
        )
        self.dipole = SimpleNamespace(dipole_moment_Cm=2.0e-29)  # ~0.6 e·nm

    def _omega(self, Ei_eV, Ef_eV):
        dE = float(Ei_eV - Ef_eV)
        return 0.0 if dE <= 0 else (dE * e) / hbar

    def _gamma0(self, omega, mu):
        if omega <= 0.0 or mu <= 0.0:
            return 0.0
        return (omega**3 * mu**2) / (3.0 * np.pi * epsilon_0 * hbar * c**3)

    def _purcell(self, omega):
        if self.cavity is None or omega <= 0.0:
            return 0.0
        lam = 2 * np.pi * c / omega
        Vm = float(self.cavity.Veff_um3) * 1e-18  # μm^3 -> m^3
        return (
            (3.0 / (4.0 * np.pi**2))
            * (lam / self.cavity.n) ** 3
            * (self.cavity.Q / Vm)
        )

    def test_raises_without_dipoleparams(self):
        with self.assertRaises(ValueError):
            _ = DecayModel(self.el, self.no_cavity, None, self.dt)

    def test_compute_free_space_no_cavity_matches_formula(self):
        model = DecayModel(self.el, self.no_cavity, self.dipole, self.dt)
        gammas = model.compute()

        mu = self.dipole.dipole_moment_Cm

        # Expected values per line (no Purcell, just γ0 * dt)
        expected = {}
        for key, (Ei, Ef) in {
            "L_XX_X1": (self.el["XX"], self.el["X1"]),
            "L_XX_X2": (self.el["XX"], self.el["X2"]),
            "L_X1_G": (self.el["X1"], self.el["G"]),
            "L_X2_G": (self.el["X2"], self.el["G"]),
        }.items():
            w = self._omega(Ei, Ef)
            g0 = self._gamma0(w, mu)
            expected[key] = g0 * self.dt

        for k in expected:
            np.testing.assert_allclose(
                gammas[k], expected[k], rtol=1e-10, atol=0.0
            )

    def test_compute_with_cavity_is_enhanced_vs_free_space(self):
        model_free = DecayModel(self.el, self.no_cavity, self.dipole, self.dt)
        model_cav = DecayModel(self.el, self.cavity, self.dipole, self.dt)

        g_free = model_free.compute()
        g_cav = model_cav.compute()

        # For all transitions with nonzero ω, cavity rates should be strictly larger
        for key in g_free:
            if g_free[key] > 0.0:
                self.assertGreater(g_cav[key], g_free[key])

    def test_zero_dipole_moment_yields_zero_rates(self):
        zero_mu = SimpleNamespace(dipole_moment_Cm=0.0)
        model = DecayModel(self.el, self.cavity, zero_mu, self.dt)
        g = model.compute()
        for val in g.values():
            self.assertEqual(val, 0.0)

    def test_non_positive_energy_gaps_give_zero_rate(self):
        # Make one downward difference zero or negative: set XX <= X1
        el_bad = dict(self.el)
        el_bad["XX"] = el_bad["X1"]  # zero gap for XX->X1

        model = DecayModel(el_bad, self.no_cavity, self.dipole, self.dt)
        g = model.compute()

        # L_XX_X1 must be zero; others remain positive
        self.assertEqual(g["L_XX_X1"], 0.0)
        self.assertGreater(g["L_XX_X2"], 0.0)
        self.assertGreater(g["L_X1_G"], 0.0)
        self.assertGreater(g["L_X2_G"], 0.0)


if __name__ == "__main__":
    unittest.main()
