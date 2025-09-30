import math
import unittest
import numpy as np

from bec.params.energy_levels import EnergyLevels  # adjust path if needed
from bec.params.transitions import Transition, TransitionType
from bec.light import LightMode


class TestEnergyLevelsBasics(unittest.TestCase):
    def test_derived_levels_and_transitions(self):
        # Choose simple numbers; disable guard so we can inspect values
        el = EnergyLevels(
            biexciton=1.9, exciton=1.0, fss=0.2, enforce_2g_guard=False
        )

        self.assertAlmostEqual(el.X1, 1.1, places=12)
        self.assertAlmostEqual(el.X2, 0.9, places=12)
        self.assertAlmostEqual(el.XX, 1.9, places=12)
        self.assertAlmostEqual(el.binding_energy, 0.1, places=12)

        # Compare tuple fields individually (float with tolerance)
        e, tr, lab = el.e_G_X1
        self.assertAlmostEqual(e, 1.1, places=12)
        self.assertEqual(tr, Transition.G_X1)
        self.assertEqual(lab, "G_X1")

        e, tr, lab = el.e_G_X2
        self.assertAlmostEqual(e, 0.9, places=12)
        self.assertEqual(tr, Transition.G_X2)
        self.assertEqual(lab, "G_X2")

        e, tr, lab = el.e_X1_XX
        self.assertAlmostEqual(e, 0.8, places=12)  # 1.9 - 1.1
        self.assertEqual(tr, Transition.X1_XX)
        self.assertEqual(lab, "X1_XX")

        e, tr, lab = el.e_X2_XX
        self.assertAlmostEqual(e, 1.0, places=12)  # 1.9 - 0.9
        self.assertEqual(tr, Transition.X2_XX)
        self.assertEqual(lab, "X2_XX")

        e, tr, lab = el.e_G_X
        self.assertAlmostEqual(e, 0.9, places=12)
        self.assertEqual(tr, Transition.G_X)
        self.assertEqual(lab, "G_X")

        e, tr, lab = el.e_X_XX
        self.assertAlmostEqual(e, 0.8, places=12)
        self.assertEqual(tr, Transition.X_XX)
        self.assertEqual(lab, "X_XX")

    def test_guard_min_binding_energy(self):
        # binding_energy = (2*exciton - biexciton)
        # Make it zero to force a failure vs default 1.0 meV threshold
        with self.assertRaises(Exception):
            _ = EnergyLevels(
                biexciton=2.0, exciton=1.0, fss=0.0, enforce_2g_guard=True
            )

    def test_guard_with_sigma_bandwidth_floor(self):
        # For sigma_t = 1e-12 s, floor ~ 6*hbar/sigma_t in meV ~ 3.95 meV.
        # Choose |E_bind| ~ 1.0 meV to be below the floor and trigger the guard.
        # E_bind = 2*exciton - biexciton (in eV). 1.0 meV = 0.001 eV.
        exciton = 1.000000
        biexciton = 1.998999  # 2*1.000000 - 1.998999 = 0.001001 eV = 1.001 meV
        with self.assertRaises(Exception):
            _ = EnergyLevels(
                biexciton=biexciton,
                exciton=exciton,
                fss=0.0,
                enforce_2g_guard=True,
                pulse_sigma_t_s=1e-12,
            )


class TestEnergyLevelsComputeModes(unittest.TestCase):
    def test_compute_modes_fss_zero(self):
        el = EnergyLevels(
            biexciton=2.0, exciton=1.0, fss=0.0, enforce_2g_guard=False
        )
        modes = el.compute_modes()
        # Degenerate case -> two modes: G<->X and X<->XX
        self.assertEqual(len(modes), 2)
        self.assertTrue(all(isinstance(m, LightMode) for m in modes))
        self.assertEqual(modes[0].transition, Transition.G_X)
        self.assertEqual(modes[1].transition, Transition.X_XX)
        self.assertEqual(modes[0].source, TransitionType.INTERNAL)
        self.assertEqual(modes[1].source, TransitionType.INTERNAL)
        # Energy checks
        self.assertAlmostEqual(modes[0].energy_ev, el.e_G_X[0], places=12)
        self.assertAlmostEqual(modes[1].energy_ev, el.e_X_XX[0], places=12)

    def test_compute_modes_fss_nonzero(self):
        el = EnergyLevels(
            biexciton=2.0, exciton=1.0, fss=0.2, enforce_2g_guard=False
        )
        modes = el.compute_modes()
        # Split case -> four modes
        self.assertEqual(len(modes), 4)
        transitions = [m.transition for m in modes]
        self.assertCountEqual(
            transitions,
            [
                Transition.G_X1,
                Transition.G_X2,
                Transition.X1_XX,
                Transition.X2_XX,
            ],
        )
        for m in modes:
            self.assertEqual(m.source, TransitionType.INTERNAL)


class TestExcitonRotationParams(unittest.TestCase):
    def test_delta_prime_zero_expect_basis_alignment(self):
        # With delta_prime = 0 and fss > 0, H is diagonal with entries
        # +delta/2 and -delta/2. The first eigenvalue (ascending) is -delta/2,
        # whose eigenvector is |X2>, so theta ~ pi/2 with this ordering.
        el = EnergyLevels(
            biexciton=2.0,
            exciton=1.0,
            fss=1.0,
            delta_prime=0.0,
            enforce_2g_guard=False,
        )
        theta, phi, evals = el.exciton_rotation_params()
        self.assertAlmostEqual(theta, math.pi / 2.0, places=6)
        self.assertAlmostEqual(phi, 0.0, places=6)
        self.assertAlmostEqual(evals[0], -0.5 * el.fss, places=12)
        self.assertAlmostEqual(evals[1], 0.5 * el.fss, places=12)

    def test_delta_zero_delta_prime_positive_symmetric_mixing(self):
        # delta = 0, delta_prime > 0 gives H = (dp/2) * sigma_x
        # First eigenvector is proportional to (1, -1), so |v1| = |v0|
        # and theta ~ pi/4.
        el = EnergyLevels(
            biexciton=2.0,
            exciton=1.0,
            fss=0.0,
            delta_prime=2.0,
            enforce_2g_guard=False,
        )
        theta, phi, evals = el.exciton_rotation_params()
        self.assertAlmostEqual(theta, math.pi / 4.0, places=6)
        self.assertAlmostEqual(evals[0], -0.5 * el.delta_prime, places=12)
        self.assertAlmostEqual(evals[1], 0.5 * el.delta_prime, places=12)


if __name__ == "__main__":
    unittest.main()
